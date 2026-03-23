#!/usr/bin/env python
"""
ANN Baseline for Daily SSC and Load
==============================================================================
Description:
This script models daily Suspended Sediment Concentration (SSC) using a feed-forward
Artificial Neural Network (ANN), then converts SSC to daily sediment load.

Design goals:
- Same data engineering and split strategy as other baselines
- Robust data/output path resolution
- Clean training loop with early stopping on validation R2
- Unified metrics for SSC and Load (R2, RMSE, NSE, PBIAS, Mass Error)
"""

import pathlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# --- CONFIG ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 60
GRAD_CLIP = 1.0

BASE_FEATURES = [
    "Q_m3s", "d50_mm", "rainfall_mm", "temp_C", "VS", "Soil_Moisture", "NDVI",
    "dQ", "Q_ratio", "HI_proxy", "API", "P_MA3", "P_MA7", "SPI_30",
    "month_sin", "month_cos", "days_gap", "Q_lag1"
]
TARGET_COL = "SSC_mgL"


def _locate_data_file(filename="data.xlsx"):
    base = pathlib.Path(__file__).resolve().parent
    for p in [base] + list(base.parents):
        candidate = p / "data" / filename
        if candidate.exists():
            return candidate
    return base / "data" / filename


def _locate_project_root():
    base = pathlib.Path(__file__).resolve().parent
    for p in [base] + list(base.parents):
        if (p / "data").exists():
            return p
    return base


DATA_PATH = _locate_data_file("data.xlsx")
PROJECT_ROOT = _locate_project_root()
RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"Baseline_ANN_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def nse_metric(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - (np.sum((pred - obs) ** 2) / den)


def pbias_metric(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    den = np.sum(obs)
    if np.isclose(den, 0.0):
        return np.nan
    return 100.0 * np.sum(pred - obs) / den


class DataProcessor:
    def __init__(self, path: pathlib.Path):
        self.raw = pd.read_excel(path)
        self._prepare()

    @staticmethod
    def _excel_to_dt(serial):
        return datetime(1899, 12, 30) + timedelta(days=int(serial))

    @staticmethod
    def _nepali_wy(dt):
        return dt.year if dt.month >= 4 else dt.year - 1

    def _prepare(self):
        df = self.raw.copy()
        df["date_dt"] = df["date"].apply(self._excel_to_dt)
        df = df.sort_values("date_dt").reset_index(drop=True)

        df["VS"] = (df["Q_m3s"] * df["slope"]) / (df["width_m"] * df["depth_m"])
        for col in ("Soil_Moisture", "NDVI"):
            if col not in df.columns:
                df[col] = 0.0

        q = df["Q_m3s"]
        df["dQ"] = q.diff().fillna(0.0)
        q7 = q.rolling(7, min_periods=1).mean()
        df["Q_ratio"] = (q / q7.replace(0, np.nan)).fillna(1.0)
        df["Q_lag1"] = q.shift(1).ffill().fillna(0.0)

        p = df["rainfall_mm"]
        p_arr = p.values
        api_vals = np.zeros(len(df))
        for k in range(1, 31):
            shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
            api_vals += (0.9 ** k) * shifted
        df["API"] = api_vals
        df["P_MA3"] = p.rolling(3, min_periods=1).mean()
        df["P_MA7"] = p.rolling(7, min_periods=1).mean()

        p30 = p.rolling(30, min_periods=5).sum()
        p30mu = p30.expanding(min_periods=5).mean()
        p30std = p30.expanding(min_periods=5).std().replace(0, np.nan)
        df["SPI_30"] = ((p30 - p30mu) / p30std).fillna(0.0)

        q_maxroll = q.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (df["dQ"] / q_maxroll).fillna(0.0)

        df["month_sin"] = np.sin(2 * np.pi * df["date_dt"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date_dt"].dt.month / 12)
        df["days_gap"] = df["date_dt"].diff().dt.days.fillna(1.0)

        df["SSC_lag1"] = df[TARGET_COL].shift(1).ffill().fillna(0.0)
        df["SSC_lag3"] = df[TARGET_COL].shift(3).ffill().fillna(0.0)

        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df["WY"] = df["date_dt"].apply(self._nepali_wy)
        self.df = df
        self.wys = sorted(df["WY"].unique())

    def split_and_scale(self, features):
        wys = self.wys
        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]

        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val = self.df[self.df["WY"] == val_wy].copy()
        test = self.df[self.df["WY"] == test_wy].copy()

        x_scaler, y_scaler = RobustScaler(), RobustScaler()

        x_tr = x_scaler.fit_transform(train[features].values)
        x_va = x_scaler.transform(val[features].values)
        x_te = x_scaler.transform(test[features].values)

        y_tr = y_scaler.fit_transform(train[[TARGET_COL]].values).ravel()
        y_va = y_scaler.transform(val[[TARGET_COL]].values).ravel()
        y_te = y_scaler.transform(test[[TARGET_COL]].values).ravel()

        return {
            "X": {"train": x_tr, "val": x_va, "test": x_te},
            "y": {"train": y_tr, "val": y_va, "test": y_te},
            "y_scaler": y_scaler,
            "dates": {
                "train": train["date_dt"].values,
                "val": val["date_dt"].values,
                "test": test["date_dt"].values,
            },
            "q_m3s": {
                "train": train["Q_m3s"].values,
                "val": val["Q_m3s"].values,
                "test": test["Q_m3s"].values,
            },
        }


class ANNRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def inverse_scale(y_scaled, y_scaler):
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def train_model(model, data):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loaders = {
        "train": make_loader(data["X"]["train"], data["y"]["train"], shuffle=True),
        "val": make_loader(data["X"]["val"], data["y"]["val"], shuffle=False),
    }

    best_state = None
    best_val_r2 = -np.inf
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.huber_loss(pred, yb, delta=0.8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy().ravel()
                val_preds.append(pred)
                val_true.append(yb.numpy().ravel())

        pred_val_s = np.concatenate(val_preds)
        true_val_s = np.concatenate(val_true)
        pred_val = inverse_scale(pred_val_s, data["y_scaler"])
        true_val = inverse_scale(true_val_s, data["y_scaler"])
        val_r2 = r2_score(true_val, pred_val)

        if val_r2 > best_val_r2 + 1e-6:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_r2


def evaluate_phase(model, data, phase):
    model.eval()
    x = torch.tensor(data["X"][phase], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().ravel()

    y_scaler = data["y_scaler"]
    p_ssc = inverse_scale(pred_scaled, y_scaler)
    a_ssc = inverse_scale(data["y"][phase], y_scaler)

    q_m3s = data["q_m3s"][phase]
    p_load = p_ssc * q_m3s * 0.0864
    a_load = a_ssc * q_m3s * 0.0864

    metrics = {
        "R2_SSC": r2_score(a_ssc, p_ssc),
        "RMSE_SSC": np.sqrt(mean_squared_error(a_ssc, p_ssc)),
        "SD_Observed_SSC": np.std(a_ssc),
        "SD_Predicted_SSC": np.std(p_ssc),
        "NSE_SSC": nse_metric(a_ssc, p_ssc),
        "PBIAS_SSC_%": pbias_metric(a_ssc, p_ssc),
        "Mass_Err_SSC_%": pbias_metric(a_ssc, p_ssc),
        "R2_Load": r2_score(a_load, p_load),
        "RMSE_Load": np.sqrt(mean_squared_error(a_load, p_load)),
        "SD_Observed_Load": np.std(a_load),
        "SD_Predicted_Load": np.std(p_load),
        "NSE_Load": nse_metric(a_load, p_load),
        "PBIAS_Load_%": pbias_metric(a_load, p_load),
        "Mass_Err_Load_%": pbias_metric(a_load, p_load),
    }

    return metrics, data["dates"][phase], a_ssc, p_ssc, a_load, p_load


def save_phase_plots(phase, dates, a_ssc, p_ssc, a_load, p_load, metrics):
    # SSC
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(dates, a_ssc, s=16, alpha=0.75, color="#111111", edgecolors="none", label="Observed SSC")
    ax.plot(dates, p_ssc, color="#0f766e", lw=2.0, label="ANN Predicted SSC")
    ax.set_title(
        f"ANN - {phase.capitalize()} SSC: R2={metrics['R2_SSC']:.3f} | NSE={metrics['NSE_SSC']:.3f} | "
        f"RMSE={metrics['RMSE_SSC']:.2f} | PBIAS={metrics['PBIAS_SSC_%']:.2f}%"
    )
    ax.set_ylabel("SSC (mg/L)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{phase.capitalize()}_SSC_Hydrograph.png", dpi=150)
    plt.close(fig)

    # Load
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(dates, a_load, s=16, alpha=0.75, color="#1d4ed8", edgecolors="none", label="Observed Load")
    ax.plot(dates, p_load, color="#166534", lw=2.0, label="ANN Predicted Load")
    ax.set_title(
        f"ANN - {phase.capitalize()} Load: R2={metrics['R2_Load']:.3f} | NSE={metrics['NSE_Load']:.3f} | "
        f"RMSE={metrics['RMSE_Load']:.2f} | PBIAS={metrics['PBIAS_Load_%']:.2f}%"
    )
    ax.set_ylabel("Sediment Load (tons/day)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{phase.capitalize()}_Load_Hydrograph.png", dpi=150)
    plt.close(fig)


def main():
    print(f"[INIT] ANN baseline initializing. Saving to {OUTPUT_DIR}")
    print(f"[PATH] Using data file: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}.")

    proc = DataProcessor(DATA_PATH)
    feats = BASE_FEATURES + ["SSC_lag1", "SSC_lag3"]
    data = proc.split_and_scale(feats)

    print(f"[RUN] Features used: {len(feats)}")
    model = ANNRegressor(in_dim=len(feats))
    model, best_val_r2 = train_model(model, data)
    print(f"[RUN] Best validation R2 during training: {best_val_r2:.4f}")

    rows = []
    for phase in ["train", "val", "test"]:
        metrics, dates, a_ssc, p_ssc, a_load, p_load = evaluate_phase(model, data, phase)
        rows.append({
            "Model": "ANN",
            "Phase": phase,
            "R2_SSC": round(metrics["R2_SSC"], 4),
            "RMSE_SSC": round(metrics["RMSE_SSC"], 2),
            "SD_Observed_SSC": round(metrics["SD_Observed_SSC"], 2),
            "SD_Predicted_SSC": round(metrics["SD_Predicted_SSC"], 2),
            "NSE_SSC": round(metrics["NSE_SSC"], 4),
            "PBIAS_SSC_%": round(metrics["PBIAS_SSC_%"], 2),
            "Mass_Err_SSC_%": round(metrics["Mass_Err_SSC_%"], 2),
            "R2_Load": round(metrics["R2_Load"], 4),
            "RMSE_Load": round(metrics["RMSE_Load"], 2),
            "SD_Observed_Load": round(metrics["SD_Observed_Load"], 2),
            "SD_Predicted_Load": round(metrics["SD_Predicted_Load"], 2),
            "NSE_Load": round(metrics["NSE_Load"], 4),
            "PBIAS_Load_%": round(metrics["PBIAS_Load_%"], 2),
            "Mass_Err_Load_%": round(metrics["Mass_Err_Load_%"], 2),
        })
        save_phase_plots(phase, dates, a_ssc, p_ssc, a_load, p_load, metrics)

    rep_df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(rep_df.to_string(index=False))
    print("=" * 110)

    rep_df.to_csv(OUTPUT_DIR / "ANN_Baseline_Metrics.csv", index=False)
    print(f"\n[DONE] Execution complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
