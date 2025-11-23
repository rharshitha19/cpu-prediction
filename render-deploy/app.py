import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load both models
models = {
    "linear": joblib.load("LinearRegression.joblib"),
    "rf": joblib.load("RandomForest.joblib")
}

class InputData(BaseModel):
    model: str
    cpu_request: float
    mem_request: float
    cpu_limit: float
    mem_limit: float
    runtime_minutes: float
    controller_kind: str

app = FastAPI()

@app.get("/")
def home():
    return {"message": "CPU Usage Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):

    # validate model choice
    if data.model not in models:
        return {"error": "Invalid model. Choose 'linear' or 'rf'."}

    selected_model = models[data.model]

    X = [[
        data.cpu_request,
        data.mem_request,
        data.cpu_limit,
        data.mem_limit,
        data.runtime_minutes,
        data.controller_kind
    ]]
    
    prediction = selected_model.predict(X)[0]
    return {
        "selected_model": data.model,
        "cpu_usage_prediction": float(prediction)
    }
