"""myapp: A Flower / PyTorch MFCC-based SpeechCommands app."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
# ============================================================
# 1) LOAD PRECOMPUTED MFCC DATASET
# ============================================================

# Auto-detect dataset path (local first → docker fallback)
LOCAL_DATA_PATH = "/Users/osman/Documents/MLOPS PROJ AUDIO FRI MERGE/myapp/myapp/mfcc_dataset_v2.pt"
DOCKER_DATA_PATH = "/app/myapp/mfcc_dataset_v2.pt"
DATA_ROOT = "/Users/osman/Documents/MLOPS PROJ AUDIO FRI MERGE/app/data/fed_iid_3clients"
DATA_ROOT_DOCKER = "/app/data/fed_iid_3clients"

# Use local if exists, otherwise use docker path
if os.path.exists(LOCAL_DATA_PATH):
    DATA_PATH = LOCAL_DATA_PATH
else:
    DATA_PATH = DOCKER_DATA_PATH
    
if os.path.exists(DATA_ROOT):
    DATA_ROOT = DATA_ROOT
else:
    DATA_ROOT = DATA_ROOT_DOCKER

# Only load data if file exists (skip in CI/testing without dataset)
X = None
y = None
classes = None
Tmax = None
NUM_CLASSES = 35  # Default for SpeechCommands
N_SAMPLES = 0

if os.path.exists(DATA_PATH):
    print(f"[MFCC] Loading {DATA_PATH} ...")
    data = torch.load(DATA_PATH)
    X = data["X"]             # shape: (N, 1, 40, 101)
    y = data["y"]             # shape: (N)
    classes = data["classes"]
    Tmax = data["Tmax"]
    NUM_CLASSES = len(classes)
    N_SAMPLES = len(X)
    print(f"[MFCC] Loaded MFCC dataset: {X.shape}, Labels: {N_SAMPLES}")
else:
    print(f"[MFCC] Dataset not found at {DATA_PATH}, skipping data load (OK for testing)")


# ============================================================
# 2) DATASET CLASS
# ============================================================

class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 3) FEDERATED DATA PARTITIONING
# ============================================================

import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

# MFCC transform (same as v2 dataset)
mfcc_tf = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 40},
)

TMAX = 101  # known from v2 dataset



# ============================================================
# Dataset class for loading WAV → MFCC
# ============================================================

class ClientMFCCDataset(Dataset):
    def __init__(self, folder, class_to_idx):
        self.samples = []
        self.class_to_idx = class_to_idx

        for cls_name in os.listdir(folder):
            cls_folder = os.path.join(folder, cls_name)
            if not os.path.isdir(cls_folder):
                continue

            for f in os.listdir(cls_folder):
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(cls_folder, f), cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, cls_name = self.samples[idx]

        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav).float()

        # mono convert if needed
        if wav.ndim > 1:
            wav = wav.mean(dim=1)

        wav = wav.unsqueeze(0)  # (1,T)

        mfcc = mfcc_tf(wav)  # (1,40,T)
        T = mfcc.shape[-1]

        if T < TMAX:
            mfcc = torch.nn.functional.pad(mfcc, (0, TMAX - T))
        else:
            mfcc = mfcc[:, :, :TMAX]

        label = self.class_to_idx[cls_name]
        return mfcc, label


# ============================================================
# Federated loader
# ============================================================

def load_data(partition_id: int, num_partitions: int):
    """
    Load strictly IID client dataset from pre-split folders.
    """

    client_folder = os.path.join(
        DATA_ROOT, f"client_{partition_id + 1}"
    )

    # Build class map
    classes = sorted(os.listdir(client_folder))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ds = ClientMFCCDataset(client_folder, class_to_idx)

    # 80/20 split
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        ds, [train_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    print(
        f"[Client {partition_id}] Loaded {len(train_ds)} train, {len(test_ds)} test samples from {client_folder}"
    )
    return train_loader, test_loader

# ============================================================
# 4) CNN MODEL
# ============================================================

class Net(nn.Module):
    """
    AudioCNN for MFCC classification.
    Input shape: (batch, 1, 40, 101)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # (1,40,101) → (64,5,12)
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# 5) TRAIN LOOP
# ============================================================

def train(net, trainloader, epochs, lr, device):
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    total_loss = 0.0

    for ep in range(epochs):
        print(f"[train] Epoch {ep+1}/{epochs}")
        for Xb, yb in trainloader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = net(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(trainloader)
    print(f"[train] Final train loss: {avg_loss}")
    return avg_loss


# ============================================================
# 6) TEST LOOP
# ============================================================

def test(net, testloader, device):
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in testloader:
            Xb, yb = Xb.to(device), yb.to(device)

            out = net(Xb)
            loss = criterion(out, yb)
            total_loss += loss.item()

            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total

    print(f"[test] loss={avg_loss:.4f} accuracy={accuracy:.4f}")
    return avg_loss, accuracy
