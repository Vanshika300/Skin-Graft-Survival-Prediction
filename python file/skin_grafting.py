# skin_grafting.py  (robust local-ready version)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# Config / paths
# -------------------------
DATA_PATH = 'Data/mini1.csv'   # change if needed
ENCODER_PATH = 'encoder.joblib'
TEMPLATE_PATH = 'template_df.pkl'
MODEL_PATH = 'best_model.h5'
FINAL_MODEL_PATH = 'final_model.h5'

# -------------------------
# Helpers
# -------------------------
def split_and_flatten(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('-')
            split_cols = df[col].str.split(',', expand=True)
            split_cols = split_cols.rename(columns=lambda x: f"{col}_{x+1}")
            df = pd.concat([df, split_cols], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df

columns_to_split = ['A', 'B', 'C', 'DR', 'A.1', 'B.1', 'C.1', 'DR.1']

def preprocess_input(input_data_dict_or_df, encoder, numerical_columns, categorical_columns, template_df):
    """
    Accepts either a dict (single sample) or a DataFrame (batch).
    Returns a DataFrame aligned to template_df columns, ready for model.predict.
    """
    if isinstance(input_data_dict_or_df, dict):
        input_df = pd.DataFrame([input_data_dict_or_df])
    else:
        input_df = input_data_dict_or_df.copy()

    input_df = split_and_flatten(input_df, columns_to_split)

    # Ensure expected categorical columns exist
    for col in categorical_columns:
        if col not in input_df.columns:
            input_df[col] = '-'
    # Cast categorical to str
    if len(categorical_columns) > 0:
        input_df[categorical_columns] = input_df[categorical_columns].astype(str)

    # Encode categorical features (if encoder provided)
    if encoder is not None and len(categorical_columns) > 0:
        try:
            encoded_arr = encoder.transform(input_df[categorical_columns])
            encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(categorical_columns), index=input_df.index)
        except Exception:
            dummies = pd.get_dummies(input_df[categorical_columns].astype(str).replace('nan','-'))
            encoder_cols = list(encoder.get_feature_names_out(categorical_columns))
            encoded_df = pd.DataFrame(0, index=input_df.index, columns=encoder_cols)
            for dummy_col in dummies.columns:
                if dummy_col in encoded_df.columns:
                    encoded_df.loc[:, dummy_col] = dummies[dummy_col].values
                else:
                    for c in categorical_columns:
                        candidate = f"{c}_{dummy_col}"
                        if candidate in encoded_df.columns:
                            encoded_df.loc[:, candidate] = dummies[dummy_col].values
    else:
        # No encoder: create empty encoded_df with zero columns
        encoded_df = pd.DataFrame(index=input_df.index)

    # Drop raw categorical and concat encoded
    input_df = input_df.drop(columns=categorical_columns, errors='ignore').reset_index(drop=True)
    if not encoded_df.empty:
        encoded_df = encoded_df.reset_index(drop=True)
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df], axis=1)

    # Ensure numeric columns exist and convert
    for col in numerical_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df[numerical_columns] = input_df[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Normalize using training means/stds from template_df
    means = template_df[numerical_columns].mean()
    stds = template_df[numerical_columns].std().replace(0, 1)
    input_df[numerical_columns] = (input_df[numerical_columns] - means) / stds

    # Align final columns with template_df (add missing cols as zeros and enforce order)
    for col in template_df.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[template_df.columns]

    return input_df

# -------------------------
# Main routine (training + sample predict)
# -------------------------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Put mini1.csv there or update DATA_PATH.")

    data = pd.read_csv(DATA_PATH)

    # Preprocess (training-time)
    data = split_and_flatten(data, columns_to_split)

    # categorical columns (exclude SURVIVAL)
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object' and col != 'SURVIVAL']

    # Fit encoder only if needed
    encoder = None
    if len(categorical_columns) > 0:
        # use sparse_output for modern sklearn
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder_arr = encoder.fit_transform(data[categorical_columns].astype(str))
        encoded_df = pd.DataFrame(encoder_arr, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)
    else:
        encoded_df = pd.DataFrame(index=data.index)

    # Drop original categorical cols and concat encoded cols
    data = data.drop(columns=categorical_columns, errors='ignore')
    if not encoded_df.empty:
        data = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # numeric handling
    numerical_columns = [col for col in data.columns if col not in ['SURVIVAL', 'MLC']]
    data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')
    means = data[numerical_columns].mean()
    stds = data[numerical_columns].std().replace(0, 1)
    data[numerical_columns] = (data[numerical_columns] - means) / stds

    # Prepare X, y and template
    template_df = data.drop(columns=['SURVIVAL'], errors='ignore').copy()
    X = template_df.copy()
    y = data['SURVIVAL'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------
    # Model
    # -------------------------
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # callbacks for safety
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                        callbacks=[early_stop, checkpoint], verbose=1)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Save artifacts
    if encoder is not None:
        joblib.dump(encoder, ENCODER_PATH)
    template_df.to_pickle(TEMPLATE_PATH)
    model.save(FINAL_MODEL_PATH)
    print("Saved artifacts:", ENCODER_PATH if encoder is not None else "(no encoder)", TEMPLATE_PATH, FINAL_MODEL_PATH)

    # -------------------------
    # Example prediction
    # -------------------------
    example_input = {
        'A': '3,9','B': '7,40','C': 'w3,-','DR': 'w2,-',
        'A.1': '1,3','B.1': '7,8','C.1': 'w1,w1','DR.1': 'w2,w3',
        'MLC': 0
    }
    pre = preprocess_input(example_input, encoder, numerical_columns, categorical_columns, template_df)
    pred = model.predict(pre)
    print("Predicted Survival:", float(pred[0][0]))

if __name__ == "__main__":
    main()
