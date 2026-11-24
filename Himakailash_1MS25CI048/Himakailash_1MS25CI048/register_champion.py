import json
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-classifier"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

with open("outputs/metrics_summary.json") as f:
    summary = json.load(f)

winner_run_id = summary["winner_run_id"]
loser_run_id = summary["loser_run_id"]

# Register Winner
winner_mv = mlflow.register_model(
    model_uri=f"runs:/{winner_run_id}/model",
    name=REGISTERED_MODEL_NAME,
)
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=winner_mv.version,
)

# Register Challenger (Loser)
loser_mv = mlflow.register_model(
    model_uri=f"runs:/{loser_run_id}/model",
    name=REGISTERED_MODEL_NAME,
)
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="challenger",
    version=loser_mv.version,
)

print(f"Registered '{REGISTERED_MODEL_NAME}':")
print(f"  @champion   -> Version {winner_mv.version} (Run ID: {winner_run_id})")
print(f"  @challenger -> Version {loser_mv.version} (Run ID: {loser_run_id})")