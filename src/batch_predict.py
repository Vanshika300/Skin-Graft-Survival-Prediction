# batch_predict.py
"""
Run batch predictions on Data/mini1.csv and save predictions_batch.csv.
"""

import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# import preprocess function + constants from predict.py
from predict import preprocess_input, COLUMNS_TO_SPLIT

# Paths
INPUT_CSV = "Data/mini1.csv"
OUT_CSV = "predictions_batch.csv"
MODEL_FP = "best_model.h5"
ENCODER_FP = "encoder.joblib"
TEMPLATE_FP = "template_df.pkl"


def interpret_val(v, t1, t2, t3):
    if v < t1:
        return "Very Poor"
    if v < t2:
        return "Poor"
    if v < t3:
        return "Moderate"
    return "Good"


def main():

    # Check files
    for f in [INPUT_CSV, MODEL_FP, ENCODER_FP, TEMPLATE_FP]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}")

    print("Loading model and artifacts...")
    model = load_model(MODEL_FP)
    encoder = joblib.load(ENCODER_FP)
    template_df = pd.read_pickle(TEMPLATE_FP)

    # derive columns
    encoder_cols = list(encoder.get_feature_names_out())
    categorical_columns = list({col.split("_")[0] for col in encoder_cols})
    numerical_columns = [c for c in template_df.columns if c not in encoder_cols]

    # Load dataset
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Preprocess
    print("Preprocessing...")
    processed = preprocess_input(df, encoder, numerical_columns, categorical_columns, template_df)

    # Predict
    print("Predicting...")
    preds = model.predict(processed).flatten()

    # Save batch results (initial)
    df_out = df.copy()
    df_out["predicted_SURVIVAL"] = preds
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved batch predictions to: {OUT_CSV}")

    # --- Interpretation using dataset quantiles (automatic) ---
        # --- Interpretation & comparison (improved) ---
    # compute thresholds from true SURVIVAL if available
    if "SURVIVAL" in df.columns:
        q25_true, q50_true, q75_true = df["SURVIVAL"].quantile([0.25, 0.5, 0.75]).tolist()
    else:
        q25_true = q50_true = q75_true = None

    # compute thresholds from predicted values
    q25_pred, q50_pred, q75_pred = np.quantile(preds, [0.25, 0.5, 0.75])

    print("\nThresholds (from TRUE SURVIVAL):", (q25_true, q50_true, q75_true))
    print("Thresholds (from PREDICTIONS):", (q25_pred, q50_pred, q75_pred))

    # label functions
    def label_by_thresholds(v, t1, t2, t3):
        if v < t1:
            return "Very Poor"
        if v < t2:
            return "Poor"
        if v < t3:
            return "Moderate"
        return "Good"

    # create predicted-labels using both approaches
    if q25_true is not None:
        df_out["predicted_label_by_true"] = [label_by_thresholds(v, q25_true, q50_true, q75_true) for v in preds]
    else:
        df_out["predicted_label_by_true"] = None  # no true-based labels available

    df_out["predicted_label_by_pred"] = [label_by_thresholds(v, q25_pred, q50_pred, q75_pred) for v in preds]

    # save final CSV (with both labels)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved batch predictions (with labels) to: {OUT_CSV}")

    # show distributions
    print("\nDistribution - labels by TRUE thresholds (counts):")
    if q25_true is not None:
        print(df_out["predicted_label_by_true"].value_counts().to_string())
    else:
        print("(no SURVIVAL column present)")

    print("\nDistribution - labels by PREDICTIONS (counts):")
    print(df_out["predicted_label_by_pred"].value_counts().to_string())

    # If true SURVIVAL exists, compare group-wise stats and show classification report
    if "SURVIVAL" in df.columns:
        print("\nGroup-wise mean SURVIVAL and predicted mean (by predicted_label_by_true):")
        grp = df_out.groupby("predicted_label_by_true").agg(
            true_mean=("SURVIVAL", "mean"),
            pred_mean=("predicted_SURVIVAL", "mean"),
            count=("SURVIVAL", "count"),
        ).reindex(["Very Poor", "Poor", "Moderate", "Good"])
        print(grp)

        # classification report: derive true labels from SURVIVAL quartiles, compare with predicted_label_by_true
        y_true_labels = df_out["predicted_label_by_true"].astype(str).values
        y_pred_labels = df_out["predicted_label_by_pred"].astype(str).values  # or use predicted_label_by_true if comparing model to itself

        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification report (predicted_label_by_pred vs true_label_by_true):")
            print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
            print("\nConfusion matrix (rows=true, cols=pred_by_pred):")
            print(confusion_matrix(y_true_labels, y_pred_labels))
        except Exception as e:
            print("Could not compute classification report (missing sklearn?). Error:", e)



if __name__ == "__main__":
    main()
