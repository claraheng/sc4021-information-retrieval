## RunClassifiers
- Timestamp: 2026-04-08 23:19:58
- Input labelled dataset: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/eval.xls

### Evaluation Summary
- Method: Random holdout evaluation using stratified train/test splits.
- Subjectivity split: 70% train / 30% test (random_state=42, stratified).
- Polarity split: 80% train / 20% test (random_state=42, stratified).
- This holdout test is your random accuracy test on the rest of the data: performance is measured on unseen test rows, not training rows.

### Subjectivity Model Metrics
- Accuracy: 0.92
- ROC-AUC: 0.7052
- Evaluated records (test set): 300
- Records classified per second: 8883.85
- End-to-end trainer runtime: 11.0805 s

Precision/Recall/F1 report:
```text
precision    recall  f1-score   support

     neutral       0.00      0.00      0.00        21
 opinionated       0.93      0.99      0.96       279

    accuracy                           0.92       300
   macro avg       0.46      0.49      0.48       300
weighted avg       0.86      0.92      0.89       300
```

### Polarity Model Metrics
- Accuracy: 0.7043010752688172
- ROC-AUC: 0.7465
- Evaluated records (test set): 186
- Records classified per second: 2385.41
- End-to-end trainer runtime: 8.5484 s

Precision/Recall/F1 report:
```text
precision    recall  f1-score   support

    negative       0.42      0.34      0.38        32
     neutral       0.76      0.88      0.81       124
    positive       0.69      0.37      0.48        30

    accuracy                           0.70       186
   macro avg       0.62      0.53      0.56       186
weighted avg       0.69      0.70      0.68       186
```

### Scalability Notes
- Throughput is reported as records classified per second from the held-out test set.
- As dataset size grows, expected prediction time scales approximately linearly with record count for fixed model and hardware.
- End-to-end runtime also includes data loading and preprocessing overhead, not only model inference.

### Discussion
- Subjectivity holdout accuracy is 0.9200; this reflects generalization to unseen rows from the same dataset distribution.
- Polarity holdout accuracy is 0.7043; polarity is typically harder because class boundaries are less distinct and labels are more imbalanced.
- ROC-AUC is 0.7052 for subjectivity and 0.7465 for polarity, indicating ranking quality beyond a single decision threshold.
- Throughput indicates good short-run scalability on this machine (8883.85 rec/s for subjectivity, 2385.41 rec/s for polarity).
- Precision/recall/F1 should be interpreted per class, especially when minority classes have lower support.

## RunPredictors
- Timestamp: 2026-04-08 23:20:30
- Input text CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/texts_to_predict.csv
- Output predictions CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/predictions.csv

### Prediction Pipeline Summary
- Stage 1: Subjectivity prediction for all input rows.
- Stage 2: Polarity prediction only for rows predicted as opinionated.

### Prediction Process Metrics
- End-to-end subjectivity predictor runtime: 13.5242 s
- End-to-end polarity predictor runtime: 7.1464 s
- Total rows in predictions file: 10368
- Opinionated rows in predictions file: 10166
- Opinionated rows predicted by polarity stage: 10166
- Rows with polarity confidence: 10166
- Mean subjectivity confidence: 0.7199
- Mean polarity confidence (predicted rows): 0.4709

### Notes
- Prediction run does not compute precision/recall/F1 by itself because labels are not available for new unlabeled input rows.
- Use the training section above (RunClassifiers) for model evaluation metrics on held-out labeled data.

## RunPredictors
- Timestamp: 2026-04-08 23:39:45
- Input text CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/texts_to_predict.csv
- Output predictions CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/predictions.csv

### Prediction Pipeline Summary
- Stage 1: Subjectivity prediction for all input rows.
- Stage 2: Polarity prediction only for rows predicted as opinionated.

### Prediction Process Metrics
- End-to-end subjectivity predictor runtime: 10.9618 s
- End-to-end polarity predictor runtime: 5.3117 s
- Total rows in predictions file: 9370
- Opinionated rows in predictions file: 9224
- Opinionated rows predicted by polarity stage: 9224
- Rows with polarity confidence: 9224
- Mean subjectivity confidence: 0.7190
- Mean polarity confidence (predicted rows): 0.4643

### Notes
- Prediction run does not compute precision/recall/F1 by itself because labels are not available for new unlabeled input rows.
- Use the training section above (RunClassifiers) for model evaluation metrics on held-out labeled data.

## RunPredictors
- Timestamp: 2026-04-08 23:45:01
- Input text CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/texts_to_predict.csv
- Output predictions CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/predictions.csv

### Prediction Pipeline Summary
- Stage 1: Subjectivity prediction for all input rows.
- Stage 2: Polarity prediction only for rows predicted as opinionated.

### Prediction Process Metrics
- End-to-end subjectivity predictor runtime: 12.1960 s
- End-to-end polarity predictor runtime: 3.8706 s
- Total rows in predictions file: 9370
- Opinionated rows in predictions file: 9224
- Opinionated rows predicted by polarity stage: 9224

### Notes
- Prediction run does not compute precision/recall/F1 by itself because labels are not available for new unlabeled input rows.
- Use the training section above (RunClassifiers) for model evaluation metrics on held-out labeled data.

## RunPredictors
- Timestamp: 2026-04-08 23:46:18
- Input text CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/texts_to_predict.csv
- Output predictions CSV: /Users/claraheng/Desktop/GitHub/sc4021-information-retrieval/classifiers/predictions.csv

### Prediction Pipeline Summary
- Stage 1: Subjectivity prediction for all input rows.
- Stage 2: Polarity prediction only for rows predicted as opinionated.

### Prediction Process Metrics
- End-to-end subjectivity predictor runtime: 12.4163 s
- End-to-end polarity predictor runtime: 6.1781 s
- Total rows in predictions file: 9370
- Opinionated rows in predictions file: 9224
- Opinionated rows predicted by polarity stage: 9224

### Notes
- Prediction run does not compute precision/recall/F1 by itself because labels are not available for new unlabeled input rows.
- Use the training section above (RunClassifiers) for model evaluation metrics on held-out labeled data.
