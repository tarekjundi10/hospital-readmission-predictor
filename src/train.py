import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from xgboost import XGBClassifier
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_and_preprocess


def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "raw", "Mental_Health_Lifestyle_Dataset.csv")
    X_train, X_test, y_train, y_test, classes = load_and_preprocess(filepath)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric="mlogloss", verbosity=0)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        report = classification_report(y_test, y_pred,
                                       target_names=classes, output_dict=True)
        auc = roc_auc_score(y_test, y_prob[:, 1])
        f1 = report["weighted avg"]["f1-score"]

        results[name] = {"model": model, "f1": f1, "auc": auc}
        print(f"{name} -- F1: {f1:.3f} | AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=classes))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"models/{name.replace(' ', '_')}_cm.png"))
        plt.close()

    # Best model
    best_name = max(results, key=lambda x: results[x]["f1"])
    best_model = results[best_name]["model"]

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        feat_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title(f"Feature Importance - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "models", "feature_importance.png"))
        plt.close()
        print("\nFeature Importance:")
        print(feat_df.to_string(index=False))

    # Model comparison
    print("\nModel Comparison:")
    print(f"{'Model':<25} {'F1':>6} {'AUC':>6}")
    print("-" * 40)
    for name, res in results.items():
        print(f"{name:<25} {res['f1']:.3f}  {res['auc']:.3f}")

    # Save best model
    model_path = os.path.join(base_dir, "models", "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name} (F1: {results[best_name]['f1']:.3f})")
    print(f"Model saved to {model_path}")

    return results, best_name


if __name__ == "__main__":
    train_and_evaluate()