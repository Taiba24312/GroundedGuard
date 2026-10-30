# RAGTruth paper: hallucination detection results (reference values)

Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., and Zhang, T.
*RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language
Models.* ACL 2024. arXiv:2401.00396v2, https://arxiv.org/abs/2401.00396

Transcribed from the arXiv v2 HTML version and checked against the arXiv v2 PDF (page 7):
every value below is identical in both. Values are percentages. The paper's test set
(Section 5.2): 450 instances, 150 per task type.

## Table 5: response-level detection (precision / recall / F1)

| method | QA P | QA R | QA F1 | Data2txt P | Data2txt R | Data2txt F1 | Summary P | Summary R | Summary F1 | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Prompt gpt-3.5-turbo | 18.8 | 84.4 | 30.8 | 65.1 | 95.5 | 77.4 | 23.4 | 89.2 | 37.1 | 37.1 | 92.3 | 52.9 |
| Prompt gpt-4-turbo | 33.2 | 90.6 | 45.6 | 64.3 | 100.0 | 78.3 | 31.5 | 97.6 | 47.6 | 46.9 | 97.9 | 63.4 |
| SelfCheckGPT gpt-3.5-turbo | 35.0 | 58.0 | 43.7 | 68.2 | 82.8 | 74.8 | 31.1 | 56.5 | 40.1 | 49.7 | 71.9 | 58.8 |
| LMvLM gpt-4-turbo | 18.7 | 76.9 | 30.1 | 68.0 | 76.7 | 72.1 | 23.3 | 81.9 | 36.2 | 36.2 | 77.8 | 49.4 |
| Finetuned Llama-2-13B | 61.6 | 76.3 | 68.2 | 85.4 | 91.0 | 88.1 | 64.0 | 54.9 | 59.1 | 76.9 | 80.7 | 78.7 |

## Table 6: span-level detection (precision / recall / F1, character level)

| method | QA P | QA R | QA F1 | Data2txt P | Data2txt R | Data2txt F1 | Summary P | Summary R | Summary F1 | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Prompt baseline gpt-3.5-turbo | 7.9 | 25.1 | 12.1 | 8.7 | 45.1 | 14.6 | 6.1 | 33.7 | 10.3 | 7.8 | 35.3 | 12.8 |
| Prompt baseline gpt-4-turbo | 23.7 | 52.0 | 32.6 | 17.9 | 66.4 | 28.2 | 14.7 | 65.4 | 24.1 | 18.4 | 60.9 | 28.3 |
| Finetuned Llama-2-13B | 55.8 | 60.8 | 58.2 | 56.5 | 50.7 | 53.5 | 52.4 | 30.8 | 38.8 | 55.6 | 50.2 | 52.7 |
