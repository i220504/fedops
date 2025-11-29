"""myapp: A Flower / PyTorch app."""
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("flower_audio_fedops")
MODEL_NAME = "Federated Audio Model"


import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from myapp.task import Net
# --------------------------
# PROMETHEUS MAX METRICS
# --------------------------
# ======================================
# PROMETHEUS MAX METRICS (INSERT HERE)
# ======================================
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, start_http_server
)

# Round basic info
fl_round = Gauge("fl_round", "Current FL round")
fl_total_rounds = Gauge("fl_total_rounds", "Total rounds planned")

# Round timing
fl_round_time = Histogram(
    "fl_round_time_seconds",
    "Time per round",
    buckets=[1,2,5,10,20,40,60,120]
)

# Train metrics
fl_train_loss = Gauge("fl_train_loss", "Training loss")
fl_train_acc = Gauge("fl_train_acc", "Training accuracy")
fl_train_samples = Gauge("fl_train_samples", "Samples used")
fl_train_time = Summary("fl_train_time_seconds", "Train time per round")

# Eval metrics
fl_eval_loss = Gauge("fl_eval_loss", "Eval loss")
fl_eval_acc = Gauge("fl_eval_acc", "Eval accuracy")
fl_eval_time = Summary("fl_eval_time_seconds", "Eval time per round")

# Client participation
fl_clients = Gauge("fl_clients", "Clients participating")
fl_clients_success = Counter("fl_clients_success", "Successful clients")
fl_clients_failed = Counter("fl_clients_failed", "Failed clients")

# Aggregation time
fl_agg_time = Histogram(
    "fl_agg_time_seconds",
    "Aggregation latency",
    buckets=[0.1,0.3,0.6,1,2,5]
)

# Model/payload sizes
fl_model_size = Gauge("fl_model_size_bytes", "Model size")
fl_payload_size = Gauge("fl_payload_size_bytes", "Payload size")

# Drift
fl_drift_score = Gauge("fl_drift_score", "Drift score")
fl_drift_flag = Gauge("fl_drift_flag", "Drift flag")
# Start Prometheus HTTP server on port 8000
start_http_server(8000)

# myapp/log_writer.py
import json, os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

from flwr.common import MetricRecord
from flwr.serverapp.strategy import FedAvg

import time

import time

class FedAvgWithLogging(FedAvg):

    def aggregate_train(self, server_round, results):
        round_start = time.time()

        # Round counter
        fl_round.set(server_round)
        fl_clients.set(len(results))

        total_samples = 0

        for msg in results:
            try:
                cid = msg.metadata.src_node_id
                m = msg.content["metrics"]

                num_ex = m.get("num-examples", 0)
                total_samples += num_ex

                fl_clients_success.inc()

            except:
                fl_clients_failed.inc()

        fl_train_samples.set(total_samples)

        # Set payload size
        # (Will be filled after super())
        
        with fl_agg_time.time():
            agg_arrays, agg_metrics = super().aggregate_train(server_round, results)

        # Update training metrics
        if agg_metrics:
            fl_train_loss.set(float(agg_metrics.get("train_loss", 0.0)))
            fl_train_acc.set(float(agg_metrics.get("train_acc", 0.0)))
            mlflow.log_metric("train_loss", float(agg_metrics.get("train_loss", 0.0)), step=server_round)
            

        fl_train_time.observe(time.time() - round_start)
        


        # Payload size
        try:
            fl_payload_size.set(agg_arrays.nbytes())
        except:
            pass

        return agg_arrays, agg_metrics


    def aggregate_evaluate(self, server_round, results):
        eval_start = time.time()

        agg_metrics = super().aggregate_evaluate(server_round, results)

        if agg_metrics:
            fl_eval_loss.set(float(agg_metrics.get("eval_loss", 0.0)))
            fl_eval_acc.set(float(agg_metrics.get("eval_acc", 0.0)))
            mlflow.log_metric("eval_loss", float(agg_metrics["eval_loss"]), step=server_round)
            mlflow.log_metric("eval_acc", float(agg_metrics["eval_acc"]), step=server_round)

        fl_eval_time.observe(time.time() - eval_start)
        
        


        return agg_metrics
import random
import math

def simulate_real_drift(server_round):
    """Realistic drift simulation using noisy, sudden, random spikes."""

    # 20% chance of a sudden spike (major drift)
    if random.random() < 0.2:
        return random.uniform(0.6, 1.0)

    # 40% chance of minor drift
    if random.random() < 0.4:
        return random.uniform(0.1, 0.4)

    # 40% chance of no drift
    return random.uniform(0.0, 0.1)

def my_callback(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    state_dict = arrays.to_torch_state_dict()
    torch.save(state_dict, f"model_round_{server_round}.pt")

    # Model size metric
    size = sum(p.nelement()*p.element_size() for p in state_dict.values())
    fl_model_size.set(size)

    # Dummy drift example (you will tune later)
    drift_score = simulate_real_drift(server_round)
    fl_drift_score.set(drift_score)
    fl_drift_flag.set(1 if drift_score > 0.5 else 0)

    mlflow.log_metric("drift_score", float(drift_score), step=server_round)
    mlflow.log_metric("drift_flag", int(drift_score > 0.5), step=server_round)



    return MetricRecord({})


RUN_STARTED = False

def write_log(data, filename="round_logs.jsonl"):
    global RUN_STARTED
    
    path = os.path.join(LOG_DIR, filename)

    # If this is the first write in the run → overwrite old file
    mode = "w" if not RUN_STARTED else "a"

    with open(path, mode) as f:
        f.write(json.dumps(data) + "\n")

    # After first write, switch to append mode
    RUN_STARTED = True




# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    
    mlflow.start_run(run_name=f"run_rounds_{context.run_config['num-server-rounds']}")
    mlflow.log_param("fraction_train", fraction_train)
    mlflow.log_param("num_rounds", num_rounds)
    mlflow.log_param("lr", lr)




    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    # strategy = FedAvg(fraction_train=fraction_train)
    strategy = FedAvgWithLogging(fraction_train=fraction_train)


    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=my_callback,
    )

    # Update Prometheus with final metrics
    # Save final model size
    final_state = result.arrays.to_torch_state_dict()
    size = sum(p.nelement()*p.element_size() for p in final_state.values())
    fl_model_size.set(size)

        


    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    mlflow.pytorch.log_model(global_model, artifact_path="model")
    
        # -----------------------------
    # Register model in Model Registry
    # -----------------------------
    print("\nRegistering model in MLflow Model Registry...")

    # Save final model as MLflow artifact
    mlflow.pytorch.log_model(global_model, artifact_path="model")

    # Register the logged model under a name
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Registered model version: {model_version.version}")

    # Set stage automatically
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging"
    )

    print(f"Model moved to Staging stage.")

    mlflow.end_run()


