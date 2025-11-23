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
DAGSHUB_MLFLOW_URI = "https://dagshub.com/rharshitha19/cpu-usage-prediction1.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("CPU_Usage_Prediction_Comparison") 
# -----------------------------

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
TARGET_COL = 'cpu_usage'
RANDOM_STATE = 42

# Ensure the 'artifacts' directory is tracked by DVC (optional, but good practice)
# You may want to run `dvc add artifacts` after the next successful repro.

def generate_and_log_shap_plots(model, X_test, model_name):
    """
    Calculates SHAP values, generates multiple SHAP plots (Summary, Dependence), 
    and logs them as artifacts to MLflow.
    """
    
    # 1. Calculate SHAP Values
    # Use a small sample for faster calculation and cleaner plots
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Ensure artifacts directory exists
    ARTIFACT_DIR = "artifacts"
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)

    # --- 2. Generate and Log SHAP Summary Plot (The plot you already have) ---
    print(f"Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plot_path_summary = os.path.join(ARTIFACT_DIR, f"shap_summary_plot_{model_name}.png")
    plt.savefig(plot_path_summary, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(plot_path_summary)
    
    # --- 3. Generate and Log SHAP Dependence Plot (Example: Top feature vs. SHAP value) ---
    # Find the index of the most important feature to use for the dependence plot
    # This feature's index is usually the first row in the summary plot's feature list
    feature_names = X_sample.columns.tolist()
    
    if feature_names:
        # Calculate mean absolute SHAP value for ranking
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_feature_index = np.argmax(mean_abs_shap)
        top_feature_name = feature_names[top_feature_index]
        
        print(f"Generating SHAP Dependence Plot for: {top_feature_name}")
        plt.figure(figsize=(8, 5))
        # Plot SHAP value of top feature vs. its actual value
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
    
    print(f"Logged all SHAP plots for {model_name} to MLflow artifacts.")


def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params, feature_cols):
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Starting model training for {model_name}...")
        model.fit(X_train, y_train)
        
        # Evaluate Metrics
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions) 
        mae = mean_absolute_error(y_test, predictions) 
        
        print(f"Metrics ({model_name} Test Set): RMSE={rmse:.4f}, R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        # --- MLflow Logging ---
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse) 
        mlflow.log_metric("mae", mae) 
        
        # Log Model artifact
        MODEL_DIR = "model"
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        MODEL_PATH = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, MODEL_PATH) 
        mlflow.log_artifact(MODEL_PATH)

        # --- SHAP PLOT GENERATION ---
        if model_name == "RandomForest":
            generate_and_log_shap_plots(model, X_test, model_name)


def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    
    # 1. Train and Log Random Forest Model
    rf_params = {"n_estimators": 150, "max_depth": 15}
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    train_and_log_model(rf_model, "RandomForest", X_train, y_train, X_test, y_test, rf_params, feature_cols)
    
    # 2. Train and Log Linear Regression Model
    lr_params = {} 
    lr_model = LinearRegression()
    train_and_log_model(lr_model, "LinearRegression", X_train, y_train, X_test, y_test, lr_params, feature_cols)
    
    # --- Write a combined metrics.json file for DVC ---
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