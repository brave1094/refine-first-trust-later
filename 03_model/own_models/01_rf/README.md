# 01_rf — Random Forest

Original algorithm: Breiman, "Random Forests" (2001). Uses the scikit-learn implementation.

- **Feature extraction**: shares as is the 1,205 statistical features of `02_xgboost/`(extractor.py + feat/xgboost_feature_a.py).
  No separate extraction code —
  when running `06_make_dataset.py --model rf`, the xgboost output is copied and reused if present,
  otherwise extracted with the same extractor.
- **Training**: `03_model/model/train_rf.py` (RandomForestClassifier,
  default n_estimators=300, supports a simple --optuna search).
- **Note**: RF does not support NaN → missing values ('-') are imputed with -1 (train_rf.py FILL_NA).
- **Importance**: impurity-based feature_importance.csv is saved to the param directory during training
  (input for the Illusion 5 feature-importance analysis; TreeSHAP analysis reuses the 09_XGBoost track tools).
