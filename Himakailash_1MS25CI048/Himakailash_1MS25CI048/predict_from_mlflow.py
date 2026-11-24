import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-classifier"
ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Fetch metadata from the registry alias to inspect version and Hugging Face model ID
mv = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=ALIAS)
run_data = client.get_run(mv.run_id).data
hf_model_id = run_data.params.get("hf_model_id", "unknown")

print(f"Loaded Registered Model : {REGISTERED_MODEL_NAME}@{ALIAS}")
print(f"Model Registry Version  : {mv.version}")
print(f"Underlying HF Model ID  : {hf_model_id}\n")

# Load model strictly via MLflow alias URI
model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}")

# Predict on brand-new unseen data
new_df = pd.read_csv("data/new_data.csv")
results_df = model.predict(new_df)

print(results_df.to_string(index=False))
results_df.to_csv("outputs/new_data_predictions.csv", index=False)
print("\nSaved predictions to outputs/new_data_predictions.csv")