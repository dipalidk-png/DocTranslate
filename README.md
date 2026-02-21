# XML-Aware Machine Translation Evaluation

This repository contains experiments for evaluating Large Language Models (LLMs) on XML-based localization machine translation tasks.

The project benchmarks multiple decoder-only models on structured XML documents while preserving markup integrity.

---

## 🚀 Models Evaluated

- **Gemma 270M**
- **Gemma 4B**
- **Sarvam Translate**

---

## 📊 Evaluation Metrics

The following metrics are computed:

- **BLEU**
- **chrF**
- **chrF++**
- **XML Tag Retention Score**

XML retention measures structural preservation of markup tags between source and predicted outputs.

---

## 🗂 Dataset

Experiments use the Salesforce XML Localization Machine Translation dataset.

⚠️ Note:
Dataset files and generated outputs are intentionally excluded from this repository.

---

## ⚙️ Installation

Create a conda environment:

```bash
conda create -n mt_eval python=3.10
conda activate mt_eval
```

Install dependencies:

```bash
pip install torch transformers datasets evaluate tqdm pandas numpy
```

---

## ▶️ Running Experiments

### Run Sarvam Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python Sarvam-Translate.py
```

### Run Gemma 270M

```bash
CUDA_VISIBLE_DEVICES=0 python gemma-3-270.py
```

### Run Gemma 4B (Multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1 python gemma-3-4B.py
```

---

## 🧪 Sanity Mode

To run a quick smoke test:

```python
SANITY_TEST = True
SANITY_SAMPLES = 100
```

For full evaluation:

```python
SANITY_TEST = False
```

---

## 📁 Repository Structure

```
.
├── Sarvam-Translate.py
├── gemma-3-270.py
├── gemma-3-4B.py
├── amta_paper_tokenizer_histogram.py
├── sample.py
└── README.md
```

---

## 🔍 Key Research Focus

- Translation quality vs model size
- Impact of long XML sequences on decoding speed
- Attention scaling in decoder-only architectures
- Structural robustness in markup-preserving MT

---

## 📈 Experimental Notes

- Inference performed with greedy decoding (`do_sample=False`)
- Mixed precision enabled (`bfloat16`)
- Multi-GPU sharding used for 4B model
- Max generation length: 512 tokens

---

## 👩‍💻 Author

**Dipali Kadam**  
Indian Institute of Technology Roorkee  

---

## 📜 License

This repository is intended for research and academic use.
