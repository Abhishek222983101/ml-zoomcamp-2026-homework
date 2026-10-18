# ML Zoomcamp 2026 – Homework

Solutions for [DataTalksClub Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).
Dataset: 2026 lead-scoring (`data/course_lead_scoring_2026.csv`), target `converted`.

```
data/                 dataset
03-classification/    Homework 3 (homework3.py, output.txt)
04-evaluation/        Homework 4 (homework4.py, output.txt, precision_recall_f1.png)
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 03-classification/homework3.py
python 04-evaluation/homework4.py
```

## Answers

### Homework 3 – Classification
| Q | Answer |
|---|--------|
| 1 Mode of industry | `technology` |
| 2 Biggest correlation | `interaction_count` and `lead_score` (0.916) |
| 3 Biggest MI | `lead_source` |
| 4 Accuracy | 0.65 |
| 5 Smallest difference | `number_of_courses_viewed` |
| 6 Best C | 0.001 |

### Homework 4 – Evaluation
| Q | Answer |
|---|--------|
| 1 Highest AUC | `lead_score` (0.788) |
| 2 Model AUC | 0.732 |
| 3 P/R intersection | 0.63 |
| 4 Max F1 threshold | 0.41 |
| 5 5-fold std | 0.007 |
| 6 Best C | 0.001 |
