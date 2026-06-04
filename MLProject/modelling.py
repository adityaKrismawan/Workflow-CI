import os
import sys
import joblib
import matplotlib.pyplot as plt
import mlflow
import dagshub
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Salary_Data_Preprocess.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
RUN_ID_FILE = ARTIFACT_DIR / "run_id.txt"

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
mlflow.autolog(log_models=False)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TRACKING_URI = Path(__file__).resolve().parent / "mlruns"
TRACKING_URI.mkdir(exist_ok=True)
TRACKING_DB = TRACKING_URI / "mlflow.db"


def configure_tracking_uri() -> tuple[str, Path]:
    explicit_uri = os.getenv("MLFLOW_TRACKING_URI")
    if explicit_uri:
        return explicit_uri, TRACKING_URI

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            dagshub_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            return dagshub_uri, TRACKING_URI
        except Exception:
            pass

    return f"sqlite:///{TRACKING_DB.as_posix()}", TRACKING_URI


def plot_feature_importance(model: RandomForestRegressor, feature_names: list[str], path: Path) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(names[:15], values[:15], color="steelblue")
    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residuals Plot")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {DATA_PATH}")

    tracking_uri, tracking_root = configure_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Salary Regression Baseline")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Salary"])
    y = df["Salary"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="baseline_random_forest"):
        run_id = mlflow.active_run().info.run_id
        RUN_ID_FILE.write_text(run_id, encoding="utf-8")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

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

        model_path = ARTIFACT_DIR / "salary_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        feature_plot = ARTIFACT_DIR / "feature_importance.png"
        plot_feature_importance(model, list(X.columns), feature_plot)
        mlflow.log_artifact(str(feature_plot), artifact_path="plots")

        residual_plot = ARTIFACT_DIR / "residuals.png"
        plot_residuals(y_test.to_numpy(), y_pred, residual_plot)
        mlflow.log_artifact(str(residual_plot), artifact_path="plots")

        summary_path = ARTIFACT_DIR / "evaluation_summary.csv"
        pd.DataFrame([
            {"rmse": rmse, "mae": mae, "r2": r2}
        ]).to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        print("MLflow run completed.")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MLflow UI: {tracking_uri}")


if __name__ == "__main__":
    main()
