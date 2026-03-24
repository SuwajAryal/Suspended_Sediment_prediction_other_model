#!/usr/bin/env python
"""
Pure Data-Driven LSTM Baseline for Daily SSC and Load
==============================================================================
Description:
This script models the daily Suspended Sediment Concentration (SSC) and 
calculates the Suspended Sediment Load (SSL) using a purely data-driven approach. 
All physical constraints (Yang's capacity, wash load floor, mass balance) have 
been completely removed to serve as a control baseline.

Key Architectural Features:
1. Sequence Length: 30-day sliding window [Batch, 30, Features].
2. Network: Plain LSTM (64 hidden) + MLP Head.
3. Loss: Standard Mean Squared Error (MSE) - no physical penalties.
"""

import warnings
import copy
import pathlib
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# --- CONFIG ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def _locate_data_file(filename="data.xlsx"):
    base = pathlib.Path(__file__).resolve().parent
    for p in [base] + list(base.parents):
        candidate = p / "data" / filename
        if candidate.exists():
            return candidate
    return base / "data" / filename

DATA_PATH = _locate_data_file("data.xlsx")
RUN_TAG    = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"Baseline_LSTM_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- HYPERPARAMETERS ---
EPOCHS       = 300
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 40
GRAD_CLIP    = 1.0
SEQ_LEN      = 30       # 30-day lookback
HUBER_DELTA  = 0.8
Q_WEIGHT_ALPHA = 1.25

# Search controls (kept optional because they increase runtime)
ENABLE_HPARAM_SEARCH = False
MAX_SEARCH_TRIALS = 16

BASE_FEATURES = [
    "Q_m3s", "Q_m3s_sq", "Q_m3s_cube", "Stream_Power_Gross",
    "d50_mm", "rainfall_mm", "temp_C", "VS", "Soil_Moisture", "NDVI",
    "dQ", "Q_ratio", "Q_lag1", "Q_lag2", "Q_lag3", "Q_max_7d", "HI_proxy",
    "API", "P_MA3", "P_MA7", "P_MA14", "P_MA30", "SPI_30"
]
TARGET_COL = "SSC_mgL"


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


def inverse_target(y_scaled, y_scaler):
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_log = y_scaler.inverse_transform(y_scaled)
    return np.expm1(y_log).flatten()


def weighted_huber_loss(pred, target, q_vals, delta=HUBER_DELTA, alpha=Q_WEIGHT_ALPHA):
    base = nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
    q_norm = q_vals / (torch.mean(q_vals) + 1e-6)
    q_norm = torch.clamp(q_norm, 0.0, 4.0)
    weights = 1.0 + alpha * q_norm
    return (base.squeeze(-1) * weights).mean()

# --- DATA PROCESSOR (LSTM Sequences) ---
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
        df["Stream_Power_Gross"] = df["Q_m3s"] * df["slope"]
        for col in ("Soil_Moisture", "NDVI"):
            if col not in df.columns: df[col] = 0.0

        Q = df["Q_m3s"]
        df["Q_m3s_sq"] = Q ** 2
        df["Q_m3s_cube"] = Q ** 3
        df["dQ"] = Q.diff().fillna(0.0)
        q7 = Q.rolling(7, min_periods=1).mean()
        df["Q_ratio"] = (Q / q7.replace(0, np.nan)).fillna(1.0)
        df["Q_lag1"] = Q.shift(1).ffill().fillna(0.0)
        df["Q_lag2"] = Q.shift(2).ffill().fillna(0.0)
        df["Q_lag3"] = Q.shift(3).ffill().fillna(0.0)
        df["Q_max_7d"] = Q.rolling(7, min_periods=1).max()

        P = df["rainfall_mm"]
        p_arr = P.values
        api_vals = np.zeros(len(df))
        for k in range(1, 31):
            shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
            api_vals += (0.9 ** k) * shifted
        df["API"] = api_vals
        df["P_MA3"] = P.rolling(3, min_periods=1).mean()
        df["P_MA7"] = P.rolling(7, min_periods=1).mean()
        df["P_MA14"] = P.rolling(14, min_periods=1).mean()
        df["P_MA30"] = P.rolling(30, min_periods=1).mean()
        
        p30 = P.rolling(30, min_periods=5).sum()
        p30mu = p30.expanding(min_periods=5).mean()
        p30std = p30.expanding(min_periods=5).std().replace(0, np.nan)
        df["SPI_30"] = ((p30 - p30mu) / p30std).fillna(0.0)

        q_val = df["Q_m3s"]
        q_delta = q_val.diff().fillna(0.0)
        q_maxroll = q_val.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (q_delta / q_maxroll).fillna(0.0)

        df["SSC_lag1"] = df[TARGET_COL].shift(1).ffill().fillna(0.0)
        df["SSC_lag2"] = df[TARGET_COL].shift(2).ffill().fillna(0.0)
        df["SSC_lag3"] = df[TARGET_COL].shift(3).ffill().fillna(0.0)
        df["SSC_lag7"] = df[TARGET_COL].shift(7).ffill().fillna(0.0)

        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df["WY"] = df["date_dt"].apply(self._nepali_wy)
        self.df = df
        self.wys = sorted(df["WY"].unique())

    def _create_sequences(self, X, y, q_vals, dates, seq_len):
        Xs, ys, qs, ds = [], [], [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:(i + seq_len)])
            ys.append(y[i + seq_len - 1])
            qs.append(q_vals[i + seq_len - 1])
            ds.append(dates[i + seq_len - 1])
        return np.array(Xs), np.array(ys), np.array(qs), np.array(ds)

    def split_and_scale(self, features, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
        wys = self.wys
        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]
        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val = self.df[self.df["WY"] == val_wy].copy()
        test = self.df[self.df["WY"] == test_wy].copy()

        x_scaler, y_scaler = RobustScaler(), RobustScaler()
        X_tr_flat = x_scaler.fit_transform(train[features].values)
        X_va_flat = x_scaler.transform(val[features].values)
        X_te_flat = x_scaler.transform(test[features].values)
        
        y_tr_log = np.log1p(np.clip(train[[TARGET_COL]].values, a_min=0.0, a_max=None))
        y_va_log = np.log1p(np.clip(val[[TARGET_COL]].values, a_min=0.0, a_max=None))
        y_te_log = np.log1p(np.clip(test[[TARGET_COL]].values, a_min=0.0, a_max=None))

        y_tr_flat = y_scaler.fit_transform(y_tr_log)
        y_va_flat = y_scaler.transform(y_va_log)
        y_te_flat = y_scaler.transform(y_te_log)

        q_tr_flat, q_va_flat, q_te_flat = train["Q_m3s"].values, val["Q_m3s"].values, test["Q_m3s"].values
        d_tr, d_va, d_te = train["date_dt"].values, val["date_dt"].values, test["date_dt"].values

        X_tr, y_tr, q_tr, dates_tr = self._create_sequences(X_tr_flat, y_tr_flat, q_tr_flat, d_tr, seq_len)
        X_va, y_va, q_va, dates_va = self._create_sequences(X_va_flat, y_va_flat, q_va_flat, d_va, seq_len)
        X_te, y_te, q_te, dates_te = self._create_sequences(X_te_flat, y_te_flat, q_te_flat, d_te, seq_len)

        return {
            "loaders": {
                "train": self.make_loader(X_tr, y_tr, q_tr, batch_size=batch_size, shuffle=True),
                "val": self.make_loader(X_va, y_va, q_va, batch_size=batch_size),
                "test": self.make_loader(X_te, y_te, q_te, batch_size=batch_size)
            },
            "y_scaler": y_scaler,
            "dates": {"train": dates_tr, "val": dates_va, "test": dates_te},
            "actuals": {
                "train": inverse_target(y_tr, y_scaler),
                "val": inverse_target(y_va, y_scaler),
                "test": inverse_target(y_te, y_scaler),
            },
            "q_m3s": {"train": q_tr, "val": q_va, "test": q_te}
        }

    @staticmethod
    def make_loader(X, y, q, batch_size=BATCH_SIZE, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(q, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# --- MODEL: OPTIMIZED LSTM ---
class LSTM_Baseline(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        
        # Attention Mechanism
        attn_weights = self.attention(out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights to sequence outputs
        context = torch.sum(attn_weights * out, dim=1)
        
        y_pred = self.head(context)
        return y_pred

# --- PURE DATA-DRIVEN TRAINING ---
def train_model(model, loaders, y_scaler, lr=LR, weight_decay=WEIGHT_DECAY):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    amp_scaler = GradScaler(enabled=USE_AMP)

    best_val_loss, best_val_r2, wait, best_state = float("inf"), -np.inf, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        for batch in loaders["train"]:
            xb, yb, qb = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                pred = model(xb)
                # Weighted robust objective improves flood-peak learning and stabilizes gradients.
                loss = weighted_huber_loss(pred, yb, qb)
                
            amp_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        val_loss = 0.0
        val_preds, val_actuals = [], []
        with torch.no_grad():
            for b in loaders["val"]:
                pred_val = model(b[0].to(DEVICE))
                val_loss += nn.functional.mse_loss(pred_val, b[1].to(DEVICE), reduction="sum").item()
                val_preds.append(pred_val.cpu().numpy())
                val_actuals.append(b[1].cpu().numpy())
        val_loss /= len(loaders["val"].dataset)
        scheduler.step(val_loss)

        val_pred_raw = inverse_target(np.vstack(val_preds), y_scaler)
        val_act_raw = inverse_target(np.vstack(val_actuals), y_scaler)
        val_r2 = r2_score(val_act_raw, val_pred_raw)

        if (val_r2 > best_val_r2 + 1e-6) or (np.isclose(val_r2, best_val_r2) and val_loss < best_val_loss):
            best_val_loss, best_val_r2, wait, best_state = val_loss, val_r2, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE: break
    
    if best_state: model.load_state_dict(best_state)
    return model, best_val_r2


def build_search_space():
    seq_lens = [21, 30, 45, 60]
    hidden_dims = [64, 96, 128]
    num_layers = [1, 2, 3]
    lrs = [3e-4, 1e-3]
    batch_sizes = [32, 64]
    dropouts = [0.1, 0.2]
    weight_decays = [5e-5, 1e-4]

    grid = []
    for cfg in itertools.product(seq_lens, hidden_dims, num_layers, lrs, batch_sizes, dropouts, weight_decays):
        grid.append({
            "seq_len": cfg[0],
            "hidden_dim": cfg[1],
            "num_layers": cfg[2],
            "lr": cfg[3],
            "batch_size": cfg[4],
            "dropout": cfg[5],
            "weight_decay": cfg[6],
        })
    return grid[:MAX_SEARCH_TRIALS]


def pick_config(proc, feats):
    if not ENABLE_HPARAM_SEARCH:
        return {
            "seq_len": SEQ_LEN,
            "hidden_dim": 128,
            "num_layers": 3,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "dropout": 0.1,
            "weight_decay": WEIGHT_DECAY,
        }

    print(f"[SEARCH] Running focused hyperparameter search across up to {MAX_SEARCH_TRIALS} trials...")
    best_cfg, best_val_r2 = None, -np.inf
    for idx, cfg in enumerate(build_search_space(), start=1):
        data_try = proc.split_and_scale(feats, seq_len=cfg["seq_len"], batch_size=cfg["batch_size"])
        model_try = LSTM_Baseline(
            in_dim=len(feats),
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        _, val_r2 = train_model(
            model_try,
            data_try["loaders"],
            data_try["y_scaler"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        print(f"        -> Trial {idx:02d}: seq={cfg['seq_len']}, hid={cfg['hidden_dim']}, layers={cfg['num_layers']}, lr={cfg['lr']}, bs={cfg['batch_size']} | Val R2={val_r2:.4f}")
        if val_r2 > best_val_r2:
            best_val_r2, best_cfg = val_r2, cfg

    print(f"[SEARCH] Best config selected with Val R2={best_val_r2:.4f}: {best_cfg}")
    return best_cfg

def main():
    print(f"[INIT] Pure Data-Driven LSTM Baseline initializing. Saving to {OUTPUT_DIR}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}.")
    proc = DataProcessor(DATA_PATH)
    
    feats = BASE_FEATURES + ["SSC_lag1", "SSC_lag2", "SSC_lag3", "SSC_lag7"]
    cfg = pick_config(proc, feats)
    print(
        f"\n[RUN] Training Baseline LSTM with {len(feats)} features | "
        f"seq={cfg['seq_len']}, hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}, "
        f"lr={cfg['lr']}, batch={cfg['batch_size']}, wd={cfg['weight_decay']}"
    )

    data = proc.split_and_scale(feats, seq_len=cfg["seq_len"], batch_size=cfg["batch_size"])
    model = LSTM_Baseline(
        in_dim=len(feats),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model, best_val_r2 = train_model(
        model,
        data["loaders"],
        data["y_scaler"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    print(f"[RUN] Best validation R2 during training: {best_val_r2:.4f}")
    
    model.eval()
    results = {}
    for phase in ["train", "val", "test"]:
        eval_loader = proc.make_loader(
            data["loaders"][phase].dataset.tensors[0].numpy(),
            data["loaders"][phase].dataset.tensors[1].numpy(),
            data["loaders"][phase].dataset.tensors[2].numpy(),
            batch_size=cfg["batch_size"],
            shuffle=False
        )
        preds = []
        with torch.no_grad():
            for b in eval_loader:
                preds.append(model(b[0].to(DEVICE)).cpu())
        
        # Calculate SSC metrics
        p_ssc = inverse_target(torch.cat(preds).numpy(), data["y_scaler"])
        a_ssc = data["actuals"][phase]
        
        # Calculate Load metrics using securely extracted Discharge (Q)
        q_m3s = data["q_m3s"][phase]
        p_load = p_ssc * q_m3s * 0.0864
        a_load = a_ssc * q_m3s * 0.0864

        r2_ssc = r2_score(a_ssc, p_ssc)
        rmse_ssc = np.sqrt(mean_squared_error(a_ssc, p_ssc))
        sd_obs_ssc = np.std(a_ssc)
        sd_pred_ssc = np.std(p_ssc)
        nse_ssc = nse_metric(a_ssc, p_ssc)
        pbias_ssc = pbias_metric(a_ssc, p_ssc)
        mass_error_ssc = (np.sum(p_ssc) - np.sum(a_ssc)) / np.sum(a_ssc) * 100
        
        r2_load = r2_score(a_load, p_load)
        rmse_load = np.sqrt(mean_squared_error(a_load, p_load))
        sd_obs_load = np.std(a_load)
        sd_pred_load = np.std(p_load)
        nse_load = nse_metric(a_load, p_load)
        pbias_load = pbias_metric(a_load, p_load)
        mass_error_load = (np.sum(p_load) - np.sum(a_load)) / np.sum(a_load) * 100
        
        results[phase] = {
            "R2_SSC": round(r2_ssc, 4), "RMSE_SSC": round(rmse_ssc, 2), "SD_Observed_SSC": round(sd_obs_ssc, 2), "SD_Predicted_SSC": round(sd_pred_ssc, 2), "NSE_SSC": round(nse_ssc, 4), "PBIAS_SSC_%": round(pbias_ssc, 2), "Mass_Error_SSC_%": round(mass_error_ssc, 2),
            "R2_Load": round(r2_load, 4), "RMSE_Load": round(rmse_load, 2), "SD_Observed_Load": round(sd_obs_load, 2), "SD_Predicted_Load": round(sd_pred_load, 2), "NSE_Load": round(nse_load, 4), "PBIAS_Load_%": round(pbias_load, 2), "Mass_Error_Load_%": round(mass_error_load, 2),
            "preds_ssc": p_ssc, "preds_load": p_load, "actuals_load": a_load
        }
        
    clean_rows = []
    for k, v in results.items():
        row = {"Phase": k}
        for metric, val in v.items():
            if metric not in ["preds_ssc", "preds_load", "actuals_load"]:
                row[metric] = val
        clean_rows.append(row)
        
    rep_df = pd.DataFrame(clean_rows)
    print("\n" + "="*80)
    print(rep_df.to_string(index=False))
    print("="*80)
    
    rep_df.to_csv(OUTPUT_DIR / "Baseline_LSTM_Metrics.csv", index=False)
    
    for phase in ["train", "val", "test"]:
        dates = data["dates"][phase]
        a_ssc = data["actuals"][phase]
        p_ssc = results[phase]["preds_ssc"]
        a_load = results[phase]["actuals_load"]
        p_load = results[phase]["preds_load"]
        
        phase_title = f"{phase.capitalize()} Phase"
        
        plt.figure(figsize=(12, 5))
        plt.scatter(dates, a_ssc, s=14, alpha=0.75, color="#111111", edgecolors="none", label="Observed SSC")
        plt.plot(dates, p_ssc, color="#0f766e", lw=2.0, label="Baseline Predicted SSC")
        plt.title(
            f"{phase_title} SSC: R²={results[phase]['R2_SSC']:.3f} | NSE={results[phase]['NSE_SSC']:.3f} | "
            f"RMSE={results[phase]['RMSE_SSC']:.2f} | PBIAS={results[phase]['PBIAS_SSC_%']:.2f}%"
        )
        plt.xlabel("Date")
        plt.ylabel("SSC (mg/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{phase_title.replace(' ', '_')}_SSC_Hydrograph.png", dpi=150)
        plt.close()
        
        plt.figure(figsize=(12, 5))
        plt.scatter(dates, a_load, s=14, alpha=0.75, color="#1d4ed8", edgecolors="none", label="Observed Load")
        plt.plot(dates, p_load, color="#166534", lw=2.0, label="Baseline Predicted Load")
        plt.title(
            f"{phase_title} Load: R²={results[phase]['R2_Load']:.3f} | NSE={results[phase]['NSE_Load']:.3f} | "
            f"RMSE={results[phase]['RMSE_Load']:.2f} | PBIAS={results[phase]['PBIAS_Load_%']:.2f}%"
        )
        plt.xlabel("Date")
        plt.ylabel("Sediment Load (tons/day)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{phase_title.replace(' ', '_')}_Load_Hydrograph.png", dpi=150)
        plt.close()

    print(f"\n[DONE] Execution complete. Results and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()