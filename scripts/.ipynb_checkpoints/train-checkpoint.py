import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    parser.add_argument("--model_type", type=str, required=True, choices=["randomforest", "xgboost", "logistic"])
    parser.add_argument("-C", "--C", type=float, default=1.0, help="Inverse of regularization strength for LogisticRegression")
    # parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    
    # --- CHANGES ARE HERE ---
    parser.add_argument("--n_estimators", type=int, default=100) # For RandomForest
    parser.add_argument("--num_round", type=int, default=100)    # For XGBoost
    # --- END CHANGES ---

    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--objective", type=str, default="binary:logistic")

    args = parser.parse_args()

    X_train = joblib.load(os.path.join(args.train, "X_train.joblib"))
    y_train = joblib.load(os.path.join(args.train, "y_train.joblib"))
    X_test = joblib.load(os.path.join(args.test, "X_test.joblib"))
    y_test = joblib.load(os.path.join(args.test, "y_test.joblib"))

    print(f"Training model: {args.model_type}")

    if args.model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    elif args.model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            # --- CHANGE IS HERE ---
            n_estimators=args.num_round, # Use the value from num_round
            # --- END CHANGE ---
            max_depth=args.max_depth,
            eta=args.eta,
            objective=args.objective,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

    elif args.model_type == "logistic":
        model = LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            solver="liblinear",  # safe choice for small/medium data
            random_state=42
        )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='binary')
    print(f"Validation F1 Score: {f1}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    main()