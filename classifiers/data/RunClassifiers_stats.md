# RunClassifiers Statistics Report

Generated on: 2026-04-04
Source log: classifiers/data/runclassifiers_output.txt

## Subjectivity Classifier

- Dataset shape: 1000 rows
- Label distribution:
  - opinionated: 929
  - neutral: 71
- Test set size (support total): 300
- Accuracy: 0.93

### Classification Summary

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| neutral | 0.00 | 0.00 | 0.00 | 21 |
| opinionated | 0.93 | 1.00 | 0.96 | 279 |
| macro avg | 0.47 | 0.50 | 0.48 | 300 |
| weighted avg | 0.86 | 0.93 | 0.90 | 300 |

### Confusion Matrix

```
[[  0  21]
 [  0 279]]
```

### Scalability

- Total records classified: 300
- Classification time: 0.0159 seconds
- Records classified per second: 18829.65

---

## Polarity Classifier

- Total rows in dataset: 1000
- Opinionated rows used for training: 929
- Label distribution:
  - neutral: 619
  - negative: 161
  - positive: 149
- Test set size (support total): 186
- Accuracy: 0.6828

### Classification Summary

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0.39 | 0.34 | 0.37 | 32 |
| neutral | 0.77 | 0.85 | 0.81 | 124 |
| positive | 0.50 | 0.33 | 0.40 | 30 |
| macro avg | 0.55 | 0.51 | 0.53 | 186 |
| weighted avg | 0.66 | 0.68 | 0.67 | 186 |

### Confusion Matrix

```
[[ 11  17   4]
 [ 12 106   6]
 [  5  15  10]]
```

### Scalability

- Total records classified: 186
- Classification time: 0.0210 seconds
- Records classified per second: 8862.11

---

## Notes

- The run completed successfully for both classifiers.
- Subjectivity logs include sklearn warnings for undefined precision on the neutral class because no samples were predicted as neutral in that test split.
