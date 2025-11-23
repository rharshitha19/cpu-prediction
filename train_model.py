import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
import mlflow
import mlflow.sklearn
import numpy as np
import joblib 
import os 
import json 
import shap 
import matplotlib.pyplot as plt 

# --- DagsHub MLflow Setup ---
# Make sure to set these in your environment:
# export MLFLOW_TRACKING_USERNAME=<your-dags-hub-username>
# export MLFLOW_TRACKING_PASSWORD=<your-personal-access-token>
DAGSHUB_MLFLOW_URI = "https://dagshub.com/rharshitha19/cpu-usage-prediction1.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

EXPERIMENT_NAME = "CPU_Usage_Prediction_Comparison"

# Check if experiment exists; create if not
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
# -----------------------------

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
TARGET_COL = 'cpu_usage'
RANDOM_STATE = 42

def generate_and_log_shap_plots(model, X_test, model_name):
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    ARTIFACT_DIR = "artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plot_path_summary = os.path.join(ARTIFACT_DIR, f"shap_summary_plot_{model_name}.png")
    plt.savefig(plot_path_summary, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(plot_path_summary)
    
    # SHAP dependence plot for top feature
    feature_names = X_sample.columns.tolist()
    if feature_names:
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_feature_index = np.argmax(mean_abs_shap)
        top_feature_name = feature_names[top_feature_index]
        
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            ind=top_feature_name, 
            shap_values=shap_values, 
            features=X_sample, 
            show=False
        )
        plot_path_dependence = os.path.join(ARTIFACT_DIR, f"shap_dependence_plot_{top_feature_name}.png")
        plt.savefig(plot_path_dependence, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(plot_path_dependence)

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params, feature_cols):
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Starting model training for {model_name}...")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"Metrics ({model_name} Test Set): RMSE={rmse:.4f}, R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        # Log parameters and metrics
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        # Save model artifact
        MODEL_DIR = "model"
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        if model_name == "RandomForest":
            generate_and_log_shap_plots(model, X_test, model_name)

def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    
    # Random Forest
    rf_params = {"n_estimators": 150, "max_depth": 15}
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    train_and_log_model(rf_model, "RandomForest", X_train, y_train, X_test, y_test, rf_params, feature_cols)
    
    # Linear Regression
    lr_params = {}
    lr_model = LinearRegression()
    train_and_log_model(lr_model, "LinearRegression", X_train, y_train, X_test, y_test, lr_params, feature_cols)
    
    # Save metrics for DVC
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)
    
    metrics_dict = {
        "RandomForest": {
            "rmse": np.sqrt(mean_squared_error(y_test, rf_predictions)),
            "r2_score": r2_score(y_test, rf_predictions),
            "mse": mean_squared_error(y_test, rf_predictions),
            "mae": mean_absolute_error(y_test, rf_predictions)
        },
        "LinearRegression": {
            "rmse": np.sqrt(mean_squared_error(y_test, lr_predictions)),
            "r2_score": r2_score(y_test, lr_predictions),
            "mse": mean_squared_error(y_test, lr_predictions),
            "mae": mean_absolute_error(y_test, lr_predictions)
        }
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

if __name__ == "__main__":
    main()
