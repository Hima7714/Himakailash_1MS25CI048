import os
import time
import json
import torch
import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- CONFIGURATION ---
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
STUDENT_USN = "YOUR_USN_HERE"  # Replace with your actual USN
SEED = 42
SUBSET_SIZE = 100  # Keep small for fast CPU execution; scale to 200 if fast enough
MARGIN = 0.02
LATENCY_GUARDRAIL_MULTIPLIER = 6.0

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "siebert/sentiment-roberta-large-english"

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# Custom MLflow PyFunc Wrapper (no direct HF calls needed in predict_from_mlflow.py)
class HFSentimentWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def load_context(self, context):
        from transformers import pipeline
        self.clf = pipeline("sentiment-analysis", model=self.model_id, device=-1)

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        raw_preds = self.clf(texts, truncation=True, max_length=256)
        return pd.DataFrame({
            "text": texts,
            "prediction": [p["label"].upper() for p in raw_preds],
            "confidence": [round(float(p["score"]), 4) for p in raw_preds],
        })


def prepare_eval_data():
    dataset = load_dataset("glue", "sst2", split="validation")
    df = dataset.to_pandas().sample(n=SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
    df["reference"] = df["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
    eval_df = df[["sentence", "reference"]].rename(columns={"sentence": "text"})
    eval_df.to_csv("data/eval_inputs.csv", index=False)
    return eval_df


def evaluate_model(run_name: str, role: str, model_id: str, eval_df: pd.DataFrame):
    with mlflow.start_run(run_name=run_name) as run:
        clf = pipeline("sentiment-analysis", model=model_id, device=-1)
        texts = eval_df["text"].tolist()

        start_time = time.perf_counter()
        raw_outputs = clf(texts, truncation=True, max_length=256)
        total_time_s = time.perf_counter() - start_time
        avg_latency_ms = (total_time_s / len(texts)) * 1000.0

        preds = [out["label"].upper() for out in raw_outputs]
        scores = [round(float(out["score"]), 4) for out in raw_outputs]

        acc = accuracy_score(eval_df["reference"], preds)
        macro_f1 = f1_score(eval_df["reference"], preds, average="macro")

        # Log Parameters
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment-classification",
            "role": role,
            "max_length": 256,
            "eval_samples": len(eval_df),
            "seed": SEED,
            "device": "cpu",
        })

        # Log Metrics
        metrics = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "total_inference_s": round(total_time_s, 2),
        }
        mlflow.log_metrics(metrics)

        # Log Tags
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "hf_model_id": model_id,
            "dataset_name": "glue/sst2-validation",
            "library_versions": f"transformers={mlflow.__version__},torch={torch.__version__}",
        })

        # Save & Log Predictions Artifact
        pred_df = eval_df.copy()
        pred_df["prediction"] = preds
        pred_df["confidence"] = scores
        pred_df["correct"] = pred_df["reference"] == pred_df["prediction"]
        pred_path = f"outputs/predictions_{role}.csv"
        pred_df.to_csv(pred_path, index=False)

        mlflow.log_artifact(pred_path)
        mlflow.log_artifact("data/eval_inputs.csv")

        # Log PyFunc Model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HFSentimentWrapper(model_id),
            input_example=pd.DataFrame({"text": ["This movie was great!"]}),
        )

        return run.info.run_id, metrics, pred_df


def create_plots(metrics_a, metrics_b, pred_a, pred_b):
    # 1. Bar Chart comparing metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = ["Model A (DistilBERT)", "Model B (RoBERTa-Large)"]

    axes[0].bar(models, [metrics_a["accuracy"], metrics_b["accuracy"]], color=["#4C72B0", "#DD8452"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy (Higher is Better)")

    axes[1].bar(models, [metrics_a["macro_f1"], metrics_b["macro_f1"]], color=["#4C72B0", "#DD8452"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Macro-F1 (Higher is Better)")

    plt.tight_layout()
    bar_path = "outputs/metrics_bar_chart.png"
    plt.savefig(bar_path)
    plt.close()

    # 2. Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["NEGATIVE", "POSITIVE"]

    cm_a = confusion_matrix(pred_a["reference"], pred_a["prediction"], labels=labels)
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Model A Confusion Matrix")

    cm_b = confusion_matrix(pred_b["reference"], pred_b["prediction"], labels=labels)
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Model B Confusion Matrix")

    plt.tight_layout()
    cm_path = "outputs/confusion_matrix_comparison.png"
    plt.savefig(cm_path)
    plt.close()

    return bar_path, cm_path


if __name__ == "__main__":
    eval_df = prepare_eval_data()

    run_id_a, metrics_a, pred_a = evaluate_model(
        "A_sentiment_distilbert-sst2", "champion_candidate", MODEL_A_ID, eval_df
    )
    run_id_b, metrics_b, pred_b = evaluate_model(
        "B_sentiment_roberta-large", "challenger_candidate", MODEL_B_ID, eval_df
    )

    bar_path, cm_path = create_plots(metrics_a, metrics_b, pred_a, pred_b)

    # Apply Pre-Defined Champion-Challenger Decision Rule
    delta_f1 = round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4)
    latency_ratio = metrics_b["avg_latency_ms"] / max(metrics_a["avg_latency_ms"], 1e-6)

    if delta_f1 >= MARGIN and latency_ratio <= LATENCY_GUARDRAIL_MULTIPLIER:
        winner = "Model B"
        winner_run_id = run_id_b
        loser_run_id = run_id_a
    else:
        winner = "Model A"
        winner_run_id = run_id_a
        loser_run_id = run_id_b

    summary_data = {
        "model_a": {"run_id": run_id_a, "model_id": MODEL_A_ID, **metrics_a},
        "model_b": {"run_id": run_id_b, "model_id": MODEL_B_ID, **metrics_b},
        "delta_macro_f1": delta_f1,
        "latency_ratio": round(latency_ratio, 2),
        "winner": winner,
        "winner_run_id": winner_run_id,
        "loser_run_id": loser_run_id,
    }
    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print 5 disagreement examples to help you write your README analysis
    disagreements = pred_a[pred_a["prediction"] != pred_b["prediction"]][["text", "reference", "prediction"]].rename(
        columns={"prediction": "model_a_pred"}
    )
    disagreements["model_b_pred"] = pred_b.loc[disagreements.index, "prediction"]
    print("\n--- SAMPLE DISAGREEMENTS (Include 5 in README) ---")
    print(disagreements.head(5).to_string(index=False))

    # Log Summary Comparison Run
    with mlflow.start_run(run_name="compare_sentiment"):
        mlflow.log_params({
            "primary_metric": "macro_f1",
            "margin": MARGIN,
            "latency_guardrail": f"<={LATENCY_GUARDRAIL_MULTIPLIER}x",
        })
        mlflow.log_metrics({
            "delta_primary_metric": delta_f1,
            "latency_ratio": round(latency_ratio, 2),
        })
        mlflow.set_tag("winner", winner)
        mlflow.log_artifact("outputs/metrics_summary.json")
        mlflow.log_artifact(bar_path)
        mlflow.log_artifact(cm_path)
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")

    print(f"\nEvaluation complete. Winner based on rule: {winner} (Delta F1: {delta_f1})")