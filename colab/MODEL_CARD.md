---
language: pt
license: other
base_model: neuralmind/bert-base-portuguese-cased
pipeline_tag: text-classification
tags:
  - misogyny-detection
  - hate-speech
  - portuguese
  - bertimbau
---

# BERTimbau fine-tuned for misogyny traits in PT-BR

Binary classifier: does a short Brazilian Portuguese social media text show traits of misogyny?
Fine-tuned from [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
It powers the [Misogyny Lens PT-BR](https://huggingface.co/spaces/hugo-guigo/misogyny-lens-ptbr) demo.

> ⚠ Research use only. The model makes mistakes and reflects biases of its training data. Do not use it to moderate or judge people.

## Results (333-sentence held-out test set, ToLD-BR, scored once)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| This model | 87.4% | 0.819 | 96.0% | 71.4% |
| Ensemble with a lexicon SVM (see the Space) | 88.0% | 0.831 | 95.2% | 73.7% |

## Training

- Data: unified corpus of HateBR, ToLD-BR and Portuguese Hate Speech, deduplicated, from the research repo [lexico-misoginia-ptbr](https://github.com/hugo-guigo/lexico-misoginia-ptbr).
- Leak control: the test sentences also appear in the unified corpus; they and their duplicates (371 rows) were removed from training.
- Balanced subsample of the train split: 11,120 texts. Max 64 tokens, lr 2e-5, batch 16, fp16, AdamW with linear warmup.
- 3 epochs on a Colab T4; epoch 2 was selected on a held-out validation split. The test set was not used for any choice.
- Label 1 = misogyny traits, label 0 = none.

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="hugo-guigo/bertimbau-misoginia-ptbr")
clf("Lugar de mulher é na cozinha")
```

## Limitations

- Small test set (333 sentences): 95% CI for accuracy is roughly ±3 points.
- Trained on Twitter-style comments; performance on news or long texts was not measured.
- Misses about 3 in 10 positive cases.

Author: Hugo Guilherme de Assis Paula (UFG). Research advised by Prof. Deborah Silva Alves Fernandes (INF/UFG). Training data follows the licenses of the source datasets.
