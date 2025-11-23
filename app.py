from flask import Flask
import os
import subprocess

app = Flask(__name__)

@app.route("/")
def run_training():
    # Run your data prep and training scripts
    subprocess.run(["python", "prepare_data.py"])
    subprocess.run(["python", "train_model.py"])
    return "Training completed successfully!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
