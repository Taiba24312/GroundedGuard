# DistilBERT-minpool vs ModernBERT-full (test, mean ± sample std over 3 seeds)

| task | metric | DistilBERT-minpool | ModernBERT-full | Δ (B − A) |
|---|---|---|---|---|
| QA | case F1 | 0.6056 ± 0.0147 | 0.6401 ± 0.0217 | +0.0345 |
| QA | span P | 0.5504 ± 0.0211 | 0.6148 ± 0.0494 | +0.0644 |
| QA | span R | 0.4620 ± 0.0095 | 0.6133 ± 0.0563 | +0.1513 |
| QA | span F1 | 0.5022 ± 0.0123 | 0.6110 ± 0.0037 | +0.1088 |
| Summary | case F1 | 0.4085 ± 0.0155 | 0.5140 ± 0.0181 | +0.1055 |
| Summary | span P | 0.3759 ± 0.0054 | 0.5194 ± 0.0440 | +0.1435 |
| Summary | span R | 0.1681 ± 0.0070 | 0.2524 ± 0.0447 | +0.0843 |
| Summary | span F1 | 0.2322 ± 0.0058 | 0.3370 ± 0.0353 | +0.1048 |
| Data2txt | case F1 | 0.8383 ± 0.0069 | 0.8694 ± 0.0129 | +0.0310 |
| Data2txt | span P | 0.4096 ± 0.0110 | 0.5017 ± 0.0424 | +0.0920 |
| Data2txt | span R | 0.4938 ± 0.0107 | 0.6269 ± 0.0556 | +0.1331 |
| Data2txt | span F1 | 0.4477 ± 0.0058 | 0.5547 ± 0.0093 | +0.1070 |
| All | case F1 | 0.7136 ± 0.0075 | 0.7592 ± 0.0115 | +0.0456 |
| All | span P | 0.4538 ± 0.0141 | 0.5447 ± 0.0411 | +0.0909 |
| All | span R | 0.4134 ± 0.0017 | 0.5429 ± 0.0531 | +0.1295 |
| All | span F1 | 0.4326 ± 0.0056 | 0.5411 ± 0.0054 | +0.1086 |

Efficiency (NVIDIA RTX PRO 6000 Blackwell Server Edition):

| | DistilBERT-minpool | ModernBERT-full |
|---|---|---|
| parameters | 66.4 M | 149.6 M |
| train time / epoch (median; runs: distilbert_minpool_v1 / modernbert_lr5e-5_s42, modernbert_lr5e-5_s43, modernbert_lr5e-5_s44) | 66.2 s | 171.2 s |
| inference / response, batch 1, median of 200 | 4.93 ms | 16.61 ms |
| peak GPU memory per training process | 5.7 GiB | 47.2 GiB |
| frozen thresholds (seed: thr) | {42: 0.91, 43: 0.9, 44: 0.9} | {42: 0.85, 43: 0.85, 44: 0.9} |

## Seed 42: char recall by label type

| scope | label type | spans | DistilBERT-minpool recall | ModernBERT-full recall | DistilBERT-minpool detected % | ModernBERT-full detected % |
|---|---|---|---|---|---|---|
| All | Evident Baseless Info | 738 | 0.4343 | 0.6508 | 59.9 | 79.0 |
| All | Evident Conflict | 619 | 0.3525 | 0.5219 | 56.4 | 72.7 |
| All | Subtle Baseless Info | 160 | 0.4781 | 0.5853 | 68.8 | 72.5 |
| All | Subtle Conflict | 16 | 0.0556 | 0.0771 | 18.8 | 18.8 |
| QA | Evident Baseless Info | 156 | 0.4808 | 0.7357 | 61.5 | 78.8 |
| QA | Evident Conflict | 30 | 0.0551 | 0.0816 | 6.7 | 13.3 |
| QA | Subtle Baseless Info | 49 | 0.4981 | 0.6693 | 71.4 | 85.7 |
| Summary | Evident Baseless Info | 117 | 0.2458 | 0.4118 | 35.0 | 64.1 |
| Summary | Evident Conflict | 100 | 0.0470 | 0.1199 | 10.0 | 19.0 |
| Summary | Subtle Baseless Info | 16 | 0.2118 | 0.2588 | 43.8 | 43.8 |
| Summary | Subtle Conflict | 11 | 0.0383 | 0.0413 | 9.1 | 9.1 |
| Data2txt | Evident Baseless Info | 465 | 0.4903 | 0.6859 | 65.6 | 82.8 |
| Data2txt | Evident Conflict | 489 | 0.5083 | 0.7345 | 68.9 | 87.3 |
| Data2txt | Subtle Baseless Info | 95 | 0.5295 | 0.5370 | 71.6 | 70.5 |
| Data2txt | Subtle Conflict | 5 | 0.0822 | 0.1324 | 40.0 | 40.0 |

## Seed 42: by number of DistilBERT windows (buckets from `outputs/module5/preds_seed42.jsonl`)

| n_windows | n | DistilBERT-minpool span P / R / F1 | ModernBERT-full span P / R / F1 |
|---|---|---|---|
| 1 | 807 | 0.4304 / 0.4279 / 0.4292 | 0.5189 / 0.6179 / 0.5641 |
| 2-3 | 1196 | 0.4502 / 0.4322 / 0.4410 | 0.4965 / 0.6670 / 0.5693 |
| 4-6 | 467 | 0.4260 / 0.4430 / 0.4343 | 0.4785 / 0.6053 / 0.5345 |
| 7-10 | 156 | 0.4202 / 0.3178 / 0.3619 | 0.5037 / 0.4191 / 0.4575 |
| >10 | 49 | 0.4208 / 0.2261 / 0.2942 | 0.5443 / 0.2137 / 0.3069 |
