# ==============================================================
#  FLASK DASHBOARD + INFERENCE + SYSTEM METRICS
# ==============================================================
import matplotlib
matplotlib.use("Agg")
from flask import Flask, render_template, jsonify, request
import json, os, io, base64, psutil
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

from myapp.task import Net   # your PyTorch model

app = Flask(__name__)

# ==============================================================
#  PATHS
# ==============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/osman/Documents/MLOPS PROJ AUDIO FRI MERGE/myapp/final_model.pt")
LOG_DIR = os.getenv("LOG_DIR", "/Users/osman/Documents/MLOPS PROJ AUDIO FRI MERGE/myapp/logs")

TRAIN_LOG = os.path.join(LOG_DIR, "dashboard_train.jsonl")
EVAL_LOG = os.path.join(LOG_DIR, "dashboard_eval.jsonl")
EVENT_LOG = os.path.join(LOG_DIR, "round_events.jsonl")
DIST_LOG = os.path.join(LOG_DIR, "data_distribution.jsonl")

# ==============================================================
#  HELPERS
# ==============================================================

def read_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ==============================================================
#  LOAD MODEL FOR INFERENCE
# ==============================================================

device = "cpu"
# ==============================================================
# LOAD CNN MFCC MODEL
# ==============================================================

from myapp.task import Net
import torchaudio
import torch.nn.functional as F
import soundfile as sf


device = "cpu"

# Same MFCC settings as training
mfcc_tf = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40},
)

TMAX = 101  # from training phase

# Load model
model = Net()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load label list (classes)
LABELS = torch.load("/Users/osman/Documents/MLOPS PROJ AUDIO FRI MERGE/mfcc_dataset_v2.pt")["classes"]



# ==============================================================
#  ROUTES — HTML Pages
# ==============================================================

@app.route("/")
def dashboard_home():
    return render_template("dashboard.html")
@app.route("/dashboard")
def dashboard_redirect():
    return render_template("dashboard.html")

@app.route("/inference")
def inference_page():
    return render_template("inference.html")

@app.route("/clients")
def clients_page():
    return render_template("clients.html")

@app.route("/system")
def system_page():
    return render_template("system.html")
@app.route("/api/distribution")
def api_distribution():
    return jsonify(read_jsonl(DIST_LOG))

# ==============================================================
#  ROUTES — FL DATA APIs
# ==============================================================
from flask import Response
import time

@app.get("/api/inference-progress")
def inference_progress():
    def generate():
        steps = [
            "Uploading audio...",
            "Converting audio to WAV...",
            "Preparing waveform...",
            "Running neural network inference...",
            "Finalizing output..."
        ]

        for i, step in enumerate(steps):
            progress = int((i + 1) / len(steps) * 100)
            msg = f"data: {json.dumps({'step': step, 'progress': progress})}\n\n"
            yield msg
            time.sleep(0.7)

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/train")
def api_train():
    return jsonify(read_jsonl(TRAIN_LOG))

@app.route("/api/eval")
def api_eval():
    return jsonify(read_jsonl(EVAL_LOG))

@app.route("/api/events")
def api_events():
    return jsonify(read_jsonl(EVENT_LOG))

# ==============================================================
#  API — SYSTEM METRICS (Prometheus-style JSON)
# ==============================================================

@app.route("/api/system")
def api_system():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    return jsonify({"cpu": cpu, "memory": mem})

# ==============================================================
#  API — REAL-TIME INFERENCE
# ==============================================================
@app.post("/api/inference")
def api_inference():
    print("\n===== NEW CNN MFCC INFERENCE =====")

    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]

    # Save temporary webm and convert
    temp_webm = "temp_input.webm"
    temp_wav = "temp_input.wav"
    audio_file.save(temp_webm)

    os.system(f"ffmpeg -loglevel quiet -y -i {temp_webm} -ar 16000 -ac 1 {temp_wav}")
    print("🎧 WAV created...")

    # ------------------------------------------------------
    # 1) LOAD RAW WAV (same as training)
    # ------------------------------------------------------
    wav_np, sr = sf.read(temp_wav)
    wav = torch.tensor(wav_np).float()

    if wav.ndim > 1:   # stereo → mono
        wav = wav.mean(dim=1)

    # ------------------------------------------------------
    # 2) SILERO VAD → remove silence
    # ------------------------------------------------------
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )

    (get_speech_ts, _, _, _, collect_chunks) = vad_utils

    speech_ts = get_speech_ts(wav, vad_model, sampling_rate=16000)

    if len(speech_ts) == 0:
        print("⚠ No speech detected. Returning UNKNOWN")
        return jsonify({"label": "Too much noise"})

    speech_wav = collect_chunks(speech_ts, wav)
    wav = speech_wav.unsqueeze(0)   # shape: (1, T)

    # ------------------------------------------------------
    # 3) MFCC (same exact preprocessing as training)
    # ------------------------------------------------------
    mfcc = mfcc_tf(wav)   # (1,40,T)

    # pad/trim to TMAX
    T = mfcc.shape[-1]
    if T < TMAX:
        mfcc = F.pad(mfcc, (0, TMAX - T))
    else:
        mfcc = mfcc[:, :, :TMAX]

    mfcc = mfcc.unsqueeze(0)  # (1,1,40,101)

    # ------------------------------------------------------
    # 4) MODEL PREDICTION + CONFIDENCE CHECK
    # ------------------------------------------------------
    with torch.no_grad():
        out = model(mfcc)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)

    conf = conf.item()
    pred = pred.item()

    THRESH = 0.3   # recommended

    if conf < THRESH:
        detected_label = "Too much noise"
    else:
        detected_label = LABELS[pred]

    print(f"🎯 Predicted: {detected_label} (conf={conf:.3f})")

    # ------------------------------------------------------
    # 5) Waveform Plot (using cleaned speech)
    # ------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    ax1.plot(wav.numpy().T, linewidth=1, color="#007bff")
    ax1.set_xticks([]); ax1.set_yticks([])
    waveform_b64 = fig_to_base64(fig1)

    # Cleanup
    os.remove(temp_webm)
    os.remove(temp_wav)

    return jsonify({"label": detected_label, "waveform": waveform_b64})

# ==============================================================
#  DEBUG
# ==============================================================

@app.get("/debug")
def debug_files():
    debug = {}

    for fname in ["dashboard_train.jsonl", "dashboard_eval.jsonl", "round_events.jsonl"]:
        fpath = os.path.join(LOG_DIR, fname)
        if not os.path.exists(fpath):
            debug[fname] = "❌ FILE NOT FOUND"
            continue

        try:
            with open(fpath, "r") as f:
                lines = f.readlines()
            debug[fname] = lines[-5:]
        except Exception as e:
            debug[fname] = f"ERROR: {e}"

    return jsonify(debug)

# ==============================================================
#  RUN
# ==============================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
