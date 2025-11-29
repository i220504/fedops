"""myapp: A Flower / PyTorch app."""
from prometheus_client import Gauge, start_http_server
import random

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from myapp.task import Net, load_data
from myapp.task import test as test_fn
from myapp.task import train as train_fn
from prometheus_client import Gauge, start_http_server


import hashlib

def tensor_hash(state_dict):
    h = hashlib.sha256()
    for k, v in state_dict.items():
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


PROM_STARTED = False

TRAIN_LOSS = Gauge("client_train_loss", "Client training loss")
EVAL_LOSS  = Gauge("client_eval_loss", "Client evaluation loss")
EVAL_ACC   = Gauge("client_eval_accuracy", "Client evaluation accuracy")


# Flower ClientApp
app = ClientApp()
# ------------------------------------------------------------
# Start Prometheus Metrics Server for this Client
# ------------------------------------------------------------


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    global PROM_STARTED
    if not PROM_STARTED:
        client_port = 8001 + context.node_config["partition-id"]
        start_http_server(client_port)
        PROM_STARTED = True

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
        # ---------------------------------------
    # CLIENT DEBUG: Print model hash received from server
    # ---------------------------------------
    received_hash = tensor_hash(model.state_dict())


    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    
    print("🔍 TRAIN METRICS SENT TO SERVER:", metrics)

    # Update Prometheus
    TRAIN_LOSS.set(train_loss)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    print("🔍 EVAL METRICS SENT TO SERVER:", metrics)

    # Update Prometheus
    EVAL_LOSS.set(eval_loss)
    EVAL_ACC.set(eval_acc)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
