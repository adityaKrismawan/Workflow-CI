import os
import sys
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "Salary_Data_Preprocess.csv"

ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

mlflow.autolog(log_models=False)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_tracking_uri():
    env_uri = os.getenv("MLFLOW_TRACKING_URI")

    if env_uri:
        return env_uri

    db_path = ROOT_DIR.parent / "mlflow.db"

    return f"sqlite:///{db_path.as_posix()}"


def plot_feature_importance(model, feature_names, save_path):
    importance = model.feature_importances_

    indices = np.argsort(importance)[::-1]

    names = [feature_names[i] for i in indices]
    values = importance[indices]

    plt.figure(figsize=(10, 6))

    plt.barh(names[:15], values[:15], color="steelblue")

    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(y_pred, residuals, alpha=0.6)

    plt.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=1,
    )

    plt.title("Residual Plot")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    tracking_uri = get_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.active_run() is None:

        mlflow.set_experiment(
            "Salary Regression Baseline"
        )

        mlflow.start_run(
            run_name="baseline_random_forest"
        )

        should_end_run = True

    else:
        should_end_run = False

    try:

        df = pd.read_csv(DATA_PATH)

        X = df.drop(columns=["Salary"])

        y = df["Salary"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
            )
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        r2 = r2_score(
            y_test,
            y_pred,
        )

        mlflow.log_params({
            "model": "RandomForestRegressor",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
            "max_depth": 10,
        })

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        current_run = mlflow.active_run()

        if current_run:

            print("=" * 60)
            print("MODEL BERHASIL DILOG")
            print(
                "RUN ID :",
                current_run.info.run_id
            )
            print(
                "TRACKING URI :",
                mlflow.get_tracking_uri()
            )
            print("=" * 60)

        model_path = ARTIFACT_DIR / "salary_model.pkl"

        joblib.dump(
            model,
            model_path,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="backup_model"
        )

        feature_plot = (
            ARTIFACT_DIR /
            "feature_importance.png"
        )

        plot_feature_importance(
            model,
            list(X.columns),
            feature_plot,
        )

        mlflow.log_artifact(
            str(feature_plot),
            artifact_path="plots"
        )

        residual_plot = (
            ARTIFACT_DIR /
            "residuals.png"
        )

        plot_residuals(
            y_test.to_numpy(),
            y_pred,
            residual_plot,
        )

        mlflow.log_artifact(
            str(residual_plot),
            artifact_path="plots"
        )

        summary_path = (
            ARTIFACT_DIR /
            "evaluation_summary.csv"
        )

        pd.DataFrame([
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        ]).to_csv(
            summary_path,
            index=False,
        )

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="reports"
        )

        print()
        print("MLflow run completed")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R2   : {r2:.4f}")
        print(
            f"Tracking URI : {tracking_uri}"
        )

    finally:

        if should_end_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()