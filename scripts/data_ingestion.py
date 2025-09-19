#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train-path", type=str, default="/opt/ml/processing/output/train")
    parser.add_argument("--output-test-path", type=str, default="/opt/ml/processing/output/test")
    parser.add_argument("--target-column", type=str, default="Churn")
    args = parser.parse_args()
    
    # Load CSV from S3
    train_df = pd.read_csv(os.path.join(args.input_data_path, "train.csv"))
    test_df  = pd.read_csv(os.path.join(args.input_data_path, "test.csv"))


    # dropping unwanted columns 
    if "CustomerID" in train_df.columns:
        train_df.drop("CustomerID", axis=1, inplace=True)
    if "CustomerID" in test_df.columns:
        test_df.drop("CustomerID", axis=1, inplace=True)
    
    #Feature Engineering
    cat_cols = ["Gender", "Subscription Type", "Contract Length"]
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        train_df[col] = enc.fit_transform(train_df[col])
        test_df[col]  = enc.transform(test_df[col])  # use same encoder
        encoders[col] = enc

    # --- Scale numeric ---
    num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"]
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols]  = scaler.transform(test_df[num_cols])  # same scaler

    X_train = train_df.drop(args.target_column, axis=1)
    y_train = train_df[args.target_column]
    X_test  = test_df.drop(args.target_column, axis=1)
    y_test  = test_df[args.target_column]

    # --- Save processed data ---
    os.makedirs(args.output_train_path, exist_ok=True)
    os.makedirs(args.output_test_path, exist_ok=True)

    joblib.dump(X_train, os.path.join(args.output_train_path, "X_train.joblib"))
    joblib.dump(y_train, os.path.join(args.output_train_path, "y_train.joblib"))
    joblib.dump(X_test,  os.path.join(args.output_test_path, "X_test.joblib"))
    joblib.dump(y_test,  os.path.join(args.output_test_path, "y_test.joblib"))
    
    X_train.to_csv(os.path.join(args.output_train_path, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_train_path, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(args.output_test_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(args.output_test_path, "y_test.csv"), index=False)


    print("Preprocessing and saving complete.")

if __name__ == "__main__":
    main()
