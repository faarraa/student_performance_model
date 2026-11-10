"""Train ANN Student Performance model and export a PyTorch .pth checkpoint."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import StudentPerformanceANN

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/an-2-an/data/master/StudentsPerformance.csv"
FEATURES = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
SCORE_COLUMNS = ["math score", "reading score", "writing score"]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data_path = Path("StudentsPerformance.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        try:
            df = pd.read_csv(DATA_URL)
        except Exception:
            import kagglehub
            import os
            downloaded = kagglehub.dataset_download("spscientist/students-performance-in-exams")
            csvs = [f for f in os.listdir(downloaded) if f.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError("CSV dataset tidak ditemukan. Simpan StudentsPerformance.csv di folder project.")
            df = pd.read_csv(os.path.join(downloaded, csvs[0]))

    df["average_score"] = df[SCORE_COLUMNS].mean(axis=1)
    df["low_performance"] = (df["average_score"] < 60).astype(int)

    categories = {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in FEATURES}
    encoded = [pd.get_dummies(df[col].astype(str)).reindex(columns=categories[col], fill_value=0).to_numpy(dtype=np.float32) for col in FEATURES]
    X = np.concatenate(encoded, axis=1).astype(np.float32)
    y = df["low_performance"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).reshape(-1, 1))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)

    model = StudentPerformanceANN(X.shape[1])
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0005)

    for epoch in range(250):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test))).numpy().ravel()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    out = Path("model")
    out.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": int(X.shape[1]),
        "categories": categories,
        "features": FEATURES,
        "threshold": 0.5,
        "target_definition": "average_score < 60",
        "metrics": metrics,
    }, out / "student_performance_model.pth")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model tersimpan: model/student_performance_model.pth")
    print("Metrik test:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
