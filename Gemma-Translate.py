# ============================================================
# 0. IMPORTS
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
import evaluate
from lxml import etree
from pathlib import Path
import shutil

# ============================================================
# 1. SEED & ENV
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
BASE_MODELS = {
    "translategemma_4b_it": "google/translategemma-4b-it",
}

DATA_ROOT = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

# TranslateGemma requires REGIONALIZED codes: "<lang>-<COUNTRY>" or "<lang>_<COUNTRY>".
# Source English: "en-US" (NOT "en-XX" — that is not in the Jinja allowlist and
# will throw `UndefinedError: 'dict object' has no attribute 'en-XX'`).
# Target codes below are the canonical forms used in the official model card examples.
SOURCE_LANG_CODE = "en"

LANG_CODE_MAP = {
    "ende": "de",
    "enfr": "fr",
    "ennl": "nl",
    "enfi": "fi",
    "enru": "ru",
}

LANG_NAME_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

BATCH_SIZE = 4          # Must be 1: chat template does not support padding across samples
MAX_NEW_TOKENS = 512
OUTPUT_FOLDER = "salesforce_eval_outputs_TranslateGemma-4B"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# =======================
# SANITY TEST TOGGLE
# =======================
SANITY_TEST = True           # <- SET False FOR FULL RUN
SANITY_SAMPLES = 10

# ============================================================
# 3. LOAD SALESFORCE DATA
# ============================================================
def normalize_salesforce_entry(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(seg.get("text", "") for seg in v["segments"])
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def load_dev_as_test(root, lang_pair):
    base = os.path.join(root, "data", lang_pair)
    src_file = os.path.join(base, f"{lang_pair}_en_dev.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_dev.json")
    with open(src_file, encoding="utf-8") as f:
        src_json = json.load(f)
    with open(tgt_file, encoding="utf-8") as f:
        tgt_json = json.load(f)
    src_texts = [normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt_texts = [normalize_salesforce_entry(v) for v in tgt_json["text"].values()]
    return src_texts, tgt_texts

# ============================================================
# 4. PROMPT — TranslateGemma official chat template
#
# Key rules (learned from Jinja source + official model card):
#  - "type" must be "text" (not "image")
#  - "source_lang_code" and "target_lang_code" must be regionalized codes
#    like "en-US", "de-DE" that are in the model's allowlist.
#    Bare codes ("en", "de") or non-standard codes ("en-XX") are NOT valid
#    and will raise a Jinja UndefinedError at template render time.
#  - "text" holds the source sentence string
#  - Only "user" role is allowed in the messages list
# ============================================================
def build_messages(src_text: str, tgt_lang_code: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": SOURCE_LANG_CODE,  # "en-US"
                    "target_lang_code": tgt_lang_code,      # e.g. "de-DE"
                    "text": src_text,
                }
            ],
        }
    ]

# ============================================================
# 5. METRICS
# ============================================================
bleu_metric  = evaluate.load("bleu")
chrf_metric  = evaluate.load("chrf")
chrf2_metric = evaluate.load("chrf")

# ============================================================
# 6. XML METRICS (paper Appendix B)
# ============================================================
def get_xml_structure(text):
    try:
        root = etree.fromstring(f"<root>{text}</root>")
        def structure(el):
            return (el.tag, [structure(c) for c in el])
        return structure(root)
    except Exception:
        return None

def compute_xml_match(predictions, references):
    matches = sum(
        1 for p, r in zip(predictions, references)
        if get_xml_structure(p) == get_xml_structure(r)
    )
    return matches / len(predictions) if predictions else 0.0

def compute_xml_chrf(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        if get_xml_structure(pred) != get_xml_structure(ref):
            scores.append(0.0)
        else:
            scores.append(chrf_metric.compute(predictions=[pred], references=[ref], beta=1)["score"])
    return float(np.mean(scores)) if scores else 0.0

def safe_bleu(predictions, references):
    if not any(p.strip() for p in predictions):
        print("  ⚠️  WARNING: All predictions empty — BLEU = 0.0")
        return 0.0
    return bleu_metric.compute(predictions=predictions, references=references)["bleu"]

# ============================================================
# 7. MAIN LOOP
# ============================================================
all_results = []

for lang_pair in LANG_PAIRS:
    print(f"\n{'='*60}")
    print(f"  Language pair: {lang_pair.upper()}")
    print(f"{'='*60}")

    src_texts, tgt_texts = load_dev_as_test(DATA_ROOT, lang_pair)

    if SANITY_TEST:
        src_texts = src_texts[:SANITY_SAMPLES]
        tgt_texts = tgt_texts[:SANITY_SAMPLES]
        print(f"⚡ SANITY MODE: {SANITY_SAMPLES} samples")
    else:
        print(f"Full run: {len(src_texts)} samples")

    tgt_lang_code = LANG_CODE_MAP[lang_pair]
    dataset = Dataset.from_dict({"src": src_texts, "ref": tgt_texts})

    for model_key, model_name in BASE_MODELS.items():
        print(f"\n→ Model: {model_key}  ({model_name})")

        # ---- Load processor (replaces AutoTokenizer) ----
        processor = AutoProcessor.from_pretrained(model_name)
        #print(processor.chat_template)
        print("PAD ID:", processor.tokenizer.pad_token_id)
        print("EOS ID:", processor.tokenizer.eos_token_id)

        # ---- Load model (must be AutoModelForImageTextToText) ----
        # Use dtype= not torch_dtype= (new transformers API)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.float32,
            #dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()

        predictions, references = [], []

        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=f"{lang_pair}"):
            batch = dataset[i : i + BATCH_SIZE]

            for src_text, ref_text in zip(batch["src"], batch["ref"]):
                messages = build_messages(src_text, tgt_lang_code)

                # Let the official Jinja template build the full prompt
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                # Move to device only — do NOT cast dtype on inputs dict,
                # as that would corrupt integer token ID tensors
                inputs = inputs.to(model.device)

                input_len = inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    # Use minimal generation args matching the official model card.
                    # Extra flags like repetition_penalty cause early EOS on this model.
                    output_ids = model.generate(
                        **inputs,
                        #attention_mask=inputs["attention_mask"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        
                        #eos_token_id=processor.tokenizer.eos_token_id,
                        #min_new_tokens=1,            # Ensure the model doesn't stop immediately
                    )

                #print("=== FULL RAW DECODE ===")
                #print(processor.decode(output_ids[0], skip_special_tokens=False))

                #print("=== CLEAN DECODE ===")
                #print(processor.decode(output_ids[0], skip_special_tokens=True))

                #exit()
                prompt_len = inputs["input_ids"].shape[-1]
                new_tokens = output_ids[0][prompt_len:]
                decoded = processor.decode(new_tokens, skip_special_tokens=True)
              
                
                #new_tokens = output_ids[0][input_len:]
              
                predictions.append(decoded.strip())
                references.append(ref_text)

            torch.cuda.empty_cache()

        # ---- Sample check ----
        print("\n--- Sample predictions ---")
        for k in range(min(3, len(predictions))):
            print(f"  SRC : {src_texts[k]}")
            print(f"  REF : {references[k]}")
            print(f"  PRED: {predictions[k]}")
            print()

        empty_count = sum(1 for p in predictions if not p.strip())
        if empty_count:
            print(f"  ⚠️  {empty_count}/{len(predictions)} predictions are empty!")

        # ====================================================
        # METRICS
        # ====================================================
        bleu_score  = safe_bleu(predictions, references)
        chrf_score  = chrf_metric.compute(predictions=predictions, references=references, beta=1)["score"]
        chrf2_score = chrf2_metric.compute(predictions=predictions, references=references, beta=2)["score"]
        xml_match   = compute_xml_match(predictions, references)
        xml_chrf    = compute_xml_chrf(predictions, references)

        print(f"{'─'*40}")
        print(f"  BLEU        : {bleu_score * 100:.2f}")
        print(f"  chrF (β=1)  : {chrf_score:.2f}")
        print(f"  chrF++ (β=2): {chrf2_score:.2f}")
        print(f"  XML-chrF    : {xml_chrf:.2f}   ← paper primary metric")
        print(f"  XML-Match   : {xml_match * 100:.2f}%  ← paper primary metric")
        print(f"{'─'*40}")

        # ====================================================
        # SAVE
        # ====================================================
        out_jsonl = f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for s, r, p in zip(src_texts, references, predictions):
                json.dump({"src": s, "ref": r, "pred": p}, f, ensure_ascii=False)
                f.write("\n")

        row = {
            "lang_pair"  : lang_pair,
            "model"      : model_key,
            "BLEU"       : round(bleu_score * 100, 2),
            "chrF"       : round(chrf_score, 2),
            "chrF++"     : round(chrf2_score, 2),
            "XML_chrF"   : round(xml_chrf, 2),
            "XML_Match"  : round(xml_match * 100, 2),
            "sanity_mode": SANITY_TEST,
            "n_samples"  : len(predictions),
        }
        all_results.append(row)
        pd.DataFrame([row]).to_csv(
            f"{OUTPUT_FOLDER}/{lang_pair}_{model_key}_metrics.csv", index=False
        )

        del model, processor
        torch.cuda.empty_cache()

# ============================================================
# 8. COMBINED RESULTS
# ============================================================
if all_results:
    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv(f"{OUTPUT_FOLDER}/ALL_metrics_summary.csv", index=False)
    print(f"\n{'='*60}")
    print("COMBINED RESULTS:")
    print(combined_df.to_string(index=False))
    print(f"{'='*60}")

shutil.make_archive(OUTPUT_FOLDER, "zip", OUTPUT_FOLDER)
print(f"\n✅ DONE → {OUTPUT_FOLDER}.zip")