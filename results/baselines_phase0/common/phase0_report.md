# Phase 0 Baseline Report

| experiment_id | model_type | preprocessing | feature_set | val_f1 | val_recall | test_accuracy | test_precision | test_recall | test_f1 | test_auc_roc | test_pr_auc | int8_test_f1 | int8_delta_f1 | int8_size_kb | int8_export_error | metrics_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-TCN-D | tcn | filtered | kp12 | 0.9000 | 1.0000 | 0.6113 | 0.3587 | 0.8182 | 0.4987 | 0.7395 | 0.3453 | 0.7843 | -0.2856 | 152.6800 |  | results/baselines_phase0/B-TCN-D/metrics.json |