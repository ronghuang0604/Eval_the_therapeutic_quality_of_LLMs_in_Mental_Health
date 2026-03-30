# Evaluating the Therapeutic Quality of LLMs in Mental Health Contexts

Fine-tuning LLaMA 3.1 8B Instruct with qLoRA for empathetic mental health response generation, with a multi-metric evaluation framework benchmarking across LLaMA, FLAN-T5, and T5.

**Authors:** Roz Huang, Sohail Khan  
**Date:** December 2024  
**Affiliation:** UC Berkeley, Master of Information and Data Science

---

## Motivation

As LLMs increasingly serve as conversational advisors, their role in emotionally sensitive contexts — particularly mental health — demands rigorous evaluation. Most LLM benchmarks measure factual accuracy or task completion, but few assess the *emotional quality* of responses. This project asks: **How well can LLMs approximate the empathetic, non-judgmental tone of a licensed therapist?**

We evaluate baseline empathetic capabilities across three model architectures, then fine-tune LLaMA 3.1 8B Instruct using qLoRA to improve therapeutic response quality.

## Key Results

| Metric | LLaMA Baseline | LLaMA Fine-Tuned |
|--------|---------------|-----------------|
| BLEU | 0.0139 | 0.0111 |
| ROUGE-1 | 0.2675 | 0.2436 |
| ROUGE-2 | 0.0388 | 0.0333 |
| ROUGE-L | 0.1251 | 0.1267 |
| BERTScore (F1) | 0.5208 | 0.5076 |

While quantitative scores remained comparable to the baseline, the fine-tuned model produced **qualitatively stronger** responses — more empathetic, contextually grounded, and therapeutically appropriate. This gap between lexical metrics and response quality highlights a known limitation of standard NLG evaluation for open-ended, empathy-driven tasks.

## Architecture & Approach

### Baseline Evaluation

Three models were benchmarked across three prompt conditions (no prompt, minimal guidance, empathetic guidance):

- **LLaMA 3.1 8B Instruct** (decoder-only, instruction-tuned) — consistent performance across all prompt conditions, suggesting relative prompt-agnosticism
- **FLAN-T5** (encoder-decoder, instruction-tuned) — highly sensitive to prompt design; significant improvement with structured prompts
- **T5** (encoder-decoder, not instruction-tuned) — lowest performance across all metrics

### Fine-Tuning with qLoRA

We fine-tuned LLaMA 3.1 8B Instruct using **Quantized Low-Rank Adaptation (qLoRA)**:

- **Quantization:** 4-bit NF4
- **Target layers:** Query and value attention projections
- **LoRA rank:** 8
- **Scaling factor (lora_alpha):** 16
- **Dropout:** 10%
- **Training data:** 6,292 examples (75/12.5/12.5 train/val/test split)

This configuration enabled fine-tuning on consumer-grade Google Colab GPUs while maintaining strong output quality.

### Evaluation Metrics

We used a combination of word-based and embedding-based metrics to capture both surface-level and semantic alignment:

- **Word-based:** BLEU, ROUGE-1, ROUGE-2, ROUGE-L
- **Embedding-based:** BERTScore, Sentence Transformer similarity (all-mpnet-base-v2)

## Iterative Debugging: Lessons Learned

The fine-tuning process required three major iterations to resolve issues, each surfacing important lessons about decoder-only model training:

### Iteration 1 — Input-Label Misalignment

**Problem:** When trained on the full dataset (4,719 examples), the model produced incoherent outputs like `"with, that a the for is you and a to to..."`. A smaller subset (525 examples) worked fine.

**Root cause:** We initially tokenized context and response separately as `input_ids` and `labels`. This breaks loss computation in decoder-only models, which require a single unified sequence.

**Fix:** Concatenated context + response into a single sequence used for both `input_ids` and `labels`.

### Iteration 2 — Special Token Formatting

**Problem:** After fixing the alignment issue, the full training set still yielded ~70% empty responses.

**Fix:** Restructured prompts to use role-based special tokens (`<system>`, `<user>`, `<assistant>`), aligning with LLaMA's expected input format.

### Iteration 3 — Decoding Bug

**Problem:** Some examples still produced empty outputs. We hypothesized flat probability distributions — but extracting logits and applying softmax showed valid token probabilities.

**Root cause:** A bug in the response generation function was mishandling the model's output during decoding.

**Fix:** Corrected the decoding logic. The model then produced consistent responses for 100% of examples.

## Dataset

Combined from three public mental health counseling datasets:

| Dataset | Source |
|---------|--------|
| [Amod/Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) | Online counseling platforms |
| [mpingale/Mental Health Chat Dataset](https://huggingface.co/datasets/mpingale/mental-health-chat-dataset) | Mental health Q&A |
| [heliosbrahma/Mental Health Chatbot Dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) | Chatbot training data |

**Total:** 6,292 examples after preprocessing (deduplication, empty response filtering, column standardization).

Each example follows the structure:
- **Context:** A user seeking mental health advice
- **Response:** An answer crafted by a qualified psychologist

## Project Structure

```
.
├── README.md
├── data/                       # Data loading and preprocessing
├── baselines/                  # Baseline model evaluation scripts
│   ├── llama_baseline.py
│   ├── flan_t5_baseline.py
│   └── t5_baseline.py
├── fine_tuning/                # qLoRA fine-tuning pipeline
│   ├── train.py
│   └── config.py
├── evaluation/                 # Evaluation metrics (BLEU, ROUGE, BERTScore, ST)
│   └── evaluate.py
├── notebooks/                  # Exploration and analysis notebooks
├── paper.pdf                   # Full research paper
└── requirements.txt
```

> **Note:** Update this structure to match your actual repo layout.

## Setup

```bash
git clone https://github.com/YOUR-HANDLE/REPO-NAME.git
cd REPO-NAME
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- PEFT (for qLoRA)
- BitsAndBytes (for 4-bit quantization)
- evaluate (for BLEU, ROUGE)
- bert-score
- sentence-transformers

## Usage

```bash
# Run baseline evaluation
python baselines/llama_baseline.py --prompt 2

# Fine-tune with qLoRA
python fine_tuning/train.py --epochs 3 --batch_size 4

# Evaluate fine-tuned model
python evaluation/evaluate.py --model_path ./checkpoints/best
```

> **Note:** Update these commands to match your actual scripts.

## Future Work

- **Evaluation gap investigation:** The discrepancy between qualitative improvement and quantitative metric stagnation suggests standard NLG metrics may not adequately capture empathy. Developing empathy-specific evaluation metrics (potentially using LLM-as-judge approaches) is a promising direction.
- **Broader domain testing:** The therapeutic tone of mental health data may inflate empathy scores. Testing on domains where empathy must be more subtly inferred (education, workplace communication) would better measure the model's adaptability.
- **Agentic integration:** Wrapping the fine-tuned model in an agent that can ask clarifying questions, track conversational context, and escalate appropriately would move toward a more realistic therapeutic assistant.

## References

1. Chen et al. *EmotionQueen: A Benchmark for Evaluating Empathy of Large Language Models.* ACL Findings, 2024.
2. Iftikhar et al. *Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CBT.* arXiv:2409.02244, 2024.
3. Luo et al. *Assessing Empathy in Large Language Models with Real-World Physician-Patient Interactions.* arXiv:2405.16402, 2024.
4. Priyadarshana et al. *Prompt engineering for digital mental health: A short review.* Frontiers in Digital Health, 2024.

## License

This project is for academic and research purposes. The datasets used are publicly available under their respective licenses.
