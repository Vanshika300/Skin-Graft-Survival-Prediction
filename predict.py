# predict.py
import argparse
import json
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# CNN feature extractor (your working file)
from cnn_feature_extractor import extract_features


# -------------------------
# Paths (adjust if needed)
# -------------------------
MODEL_FP = "best_model.h5"        # ANN trained with (tabular + 128 CNN features)
ENCODER_FP = "encoder.joblib"
TEMPLATE_FP = "template_df.pkl"


# -------------------------
# Utilities
# -------------------------
def safe_load_tabular_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ANN model not found: {path}")
    return load_model(path)


def preprocess_input(input_df, encoder, numerical_columns, categorical_columns, template_df):
    """
    Preprocess tabular (CSV) input exactly like training:
    - encode categorical
    - normalize numerical
    - align to template_df columns
    """

    # Ensure categorical columns exist
    for col in categorical_columns:
        if col not in input_df.columns:
            input_df[col] = '-'
    if categorical_columns:
        input_df[categorical_columns] = input_df[categorical_columns].astype(str)

    # Encode categoricals
    encoded_df = pd.DataFrame(index=input_df.index)
    if encoder is not None and categorical_columns:
        try:
            enc_input = input_df[encoder.feature_names_in_].astype(str)
            enc_arr = encoder.transform(enc_input)
            enc_cols = list(encoder.get_feature_names_out())
            encoded_df = pd.DataFrame(enc_arr, columns=enc_cols, index=input_df.index)
        except Exception:
            encoded_df = pd.DataFrame(
                0, index=input_df.index, columns=encoder.get_feature_names_out()
            )

    # Drop raw categoricals and concat encoded
    input_df = input_df.drop(columns=categorical_columns, errors="ignore").reset_index(drop=True)
    if not encoded_df.empty:
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Ensure numerical columns exist
    for col in numerical_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df[numerical_columns] = input_df[numerical_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Normalize numerical columns
    means = template_df[numerical_columns].mean()
    stds = template_df[numerical_columns].std().replace(0, 1)
    input_df[numerical_columns] = (input_df[numerical_columns] - means) / stds

    # Align final columns to template_df
    for col in template_df.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[template_df.columns]

    return input_df


# -------------------------
# Main prediction logic
# -------------------------
def main(args):
    print("\nLoading ANN and preprocessing artifacts...")
    model = safe_load_tabular_model(MODEL_FP)
    encoder = joblib.load(ENCODER_FP)
    template_df = pd.read_pickle(TEMPLATE_FP)

    # Derive categorical & numerical columns
    try:
        encoder_cols = list(encoder.get_feature_names_out())
        categorical_columns = list({c.split('_')[0] for c in encoder_cols})
    except Exception:
        categorical_columns = []

    numerical_columns = [c for c in template_df.columns if c not in encoder_cols]

    # -------------------------
    # Parse JSON input
    # -------------------------
    if not args.json_input:
        print("No json_input provided.")
        return

    inp = json.loads(args.json_input)
    inp_df = pd.DataFrame([inp])

    # -------------------------
    # CNN FEATURE EXTRACTION
    # -------------------------
    if args.image and os.path.exists(args.image):
        print("Extracting CNN features from image...")
        cnn_features = extract_features(args.image)   # shape (128,)
    else:
        print("No image provided. Using zero CNN features.")
        cnn_features = np.zeros(128)

    # -------------------------
    # Tabular preprocessing
    # -------------------------
    processed_tabular = preprocess_input(
        inp_df, encoder, numerical_columns, categorical_columns, template_df
    )

    # -------------------------
    # CONCATENATE (TABULAR + CNN)
    # -------------------------
    final_features = np.concatenate(
        [processed_tabular.values, cnn_features.reshape(1, -1)],
        axis=1
    )

    # -------------------------
    # Prediction
    # -------------------------
    pred = model.predict(final_features)
    val = float(pred[0][0])

    print(f"\nPredicted SURVIVAL value: {val:.6f}")

    # -------------------------
    # Save prediction
    # -------------------------
    out_row = inp.copy()
    out_row["predicted_SURVIVAL"] = val
    out_df = pd.DataFrame([out_row])

    out_csv = "predictions.csv"
    out_df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    print("Saved prediction to:", out_csv)


# -------------------------
# CLI entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_input", type=str, help="JSON input string")
    parser.add_argument("--image", type=str, help="Path to wound image (optional)", default=None)

    # Default example input
    default_input = {
        "A": "3,9",
        "B": "7,40",
        "C": "w3,-",
        "DR": "w2,-",
        "A.1": "1,3",
        "B.1": "7,8",
        "C.1": "w1,w1",
        "DR.1": "w2,w3",
        "MLC": 0
    }

    args = parser.parse_args()
    if not args.json_input:
        args.json_input = json.dumps(default_input)

    main(args)