#!/usr/bin/env python
"""
Unified Baseline Script: XGBoost + RandomForest + SVR + ANN + LSTM
==============================================================================
This script trains all baseline models in one place using only:
- Q
- rainfall
- lag values of Q and rainfall
- SSC lags

Target: SSC_mgL
Derived load: SSC * Q * 0.0864
"""

import pathlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- CONFIG ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 30
ANN_EPOCHS = 400
LSTM_EPOCHS = 400
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 50
GRAD_CLIP = 1.0
TARGET_COL = "SSC_mgL"


def build_feature_list():
    # User-requested input set only: Q, rainfall, lag values of them, and SSC lags.
    return [
        "Q_m3s",
        "rainfall_mm",
        "Q_lag1",
        "Q_lag2",
        "Q_lag3",
        "rain_lag1",
        "rain_lag2",
        "rain_lag3",
        "SSC_lag1",
        "SSC_lag2",
    ]


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

        q = df["Q_m3s"]
        r = df["rainfall_mm"]

        df["Q_lag1"] = q.shift(1).ffill().fillna(0.0)
        df["Q_lag2"] = q.shift(2).ffill().fillna(0.0)
        df["Q_lag3"] = q.shift(3).ffill().fillna(0.0)

        df["rain_lag1"] = r.shift(1).ffill().fillna(0.0)
        df["rain_lag2"] = r.shift(2).ffill().fillna(0.0)
        df["rain_lag3"] = r.shift(3).ffill().fillna(0.0)

        df["SSC_lag1"] = df[TARGET_COL].shift(1).ffill().fillna(0.0)
        df["SSC_lag2"] = df[TARGET_COL].shift(2).ffill().fillna(0.0)

        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df["WY"] = df["date_dt"].apply(self._nepali_wy)

        self.df = df
        self.wys = sorted(df["WY"].unique())

    def split(self):
        wys = self.wys
        if len(wys) < 3:
            raise ValueError("Need at least 3 water years for train/val/test split.")

        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]
        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val = self.df[self.df["WY"] == val_wy].copy()
        test = self.df[self.df["WY"] == test_wy].copy()

        return {"train": train, "val": val, "test": test}


def prepare_flat_data(split_dfs, feature_cols):
    x_scaler, y_scaler = RobustScaler(), RobustScaler()

    x_tr = x_scaler.fit_transform(split_dfs["train"][feature_cols].values)
    x_va = x_scaler.transform(split_dfs["val"][feature_cols].values)
    x_te = x_scaler.transform(split_dfs["test"][feature_cols].values)

    y_tr = y_scaler.fit_transform(split_dfs["train"][[TARGET_COL]].values).ravel()
    y_va = y_scaler.transform(split_dfs["val"][[TARGET_COL]].values).ravel()
    y_te = y_scaler.transform(split_dfs["test"][[TARGET_COL]].values).ravel()

    return {
        "X": {"train": x_tr, "val": x_va, "test": x_te},
        "y": {"train": y_tr, "val": y_va, "test": y_te},
        "y_scaler": y_scaler,
        "dates": {
            "train": split_dfs["train"]["date_dt"].values,
            "val": split_dfs["val"]["date_dt"].values,
            "test": split_dfs["test"]["date_dt"].values,
        },
        "q_m3s": {
            "train": split_dfs["train"]["Q_m3s"].values,
            "val": split_dfs["val"]["Q_m3s"].values,
            "test": split_dfs["test"]["Q_m3s"].values,
        },
    }


def create_sequences(x, y, q, dates, seq_len):
    xs, ys, qs, ds = [], [], [], []
    for i in range(len(x) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
        qs.append(q[i + seq_len - 1])
        ds.append(dates[i + seq_len - 1])
    return np.array(xs), np.array(ys), np.array(qs), np.array(ds)


def prepare_lstm_data(split_dfs, feature_cols, seq_len=SEQ_LEN):
    x_scaler, y_scaler = RobustScaler(), RobustScaler()

    x_tr_f = x_scaler.fit_transform(split_dfs["train"][feature_cols].values)
    x_va_f = x_scaler.transform(split_dfs["val"][feature_cols].values)
    x_te_f = x_scaler.transform(split_dfs["test"][feature_cols].values)

    y_tr_f = y_scaler.fit_transform(split_dfs["train"][[TARGET_COL]].values).ravel()
    y_va_f = y_scaler.transform(split_dfs["val"][[TARGET_COL]].values).ravel()
    y_te_f = y_scaler.transform(split_dfs["test"][[TARGET_COL]].values).ravel()

    q_tr = split_dfs["train"]["Q_m3s"].values
    q_va = split_dfs["val"]["Q_m3s"].values
    q_te = split_dfs["test"]["Q_m3s"].values

    d_tr = split_dfs["train"]["date_dt"].values
    d_va = split_dfs["val"]["date_dt"].values
    d_te = split_dfs["test"]["date_dt"].values

    x_tr, y_tr, q_tr_s, d_tr_s = create_sequences(x_tr_f, y_tr_f, q_tr, d_tr, seq_len)
    x_va, y_va, q_va_s, d_va_s = create_sequences(x_va_f, y_va_f, q_va, d_va, seq_len)
    x_te, y_te, q_te_s, d_te_s = create_sequences(x_te_f, y_te_f, q_te, d_te, seq_len)

    return {
        "X": {"train": x_tr, "val": x_va, "test": x_te},
        "y": {"train": y_tr, "val": y_va, "test": y_te},
        "y_scaler": y_scaler,
        "dates": {"train": d_tr_s, "val": d_va_s, "test": d_te_s},
        "q_m3s": {"train": q_tr_s, "val": q_va_s, "test": q_te_s},
    }


def inverse_scale(y_scaled, y_scaler):
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).flatten()


def compute_metrics(a_ssc, p_ssc, q_m3s):
    p_load = p_ssc * q_m3s * 0.0864
    a_load = a_ssc * q_m3s * 0.0864

    return {
        "R2_SSC": r2_score(a_ssc, p_ssc),
        "RMSE_SSC": np.sqrt(mean_squared_error(a_ssc, p_ssc)),
        "NSE_SSC": nse_metric(a_ssc, p_ssc),
        "PBIAS_SSC_%": pbias_metric(a_ssc, p_ssc),
        "Mass_Err_SSC_%": pbias_metric(a_ssc, p_ssc),
        "R2_Load": r2_score(a_load, p_load),
        "RMSE_Load": np.sqrt(mean_squared_error(a_load, p_load)),
        "NSE_Load": nse_metric(a_load, p_load),
        "PBIAS_Load_%": pbias_metric(a_load, p_load),
        "Mass_Err_Load_%": pbias_metric(a_load, p_load),
        "a_load": a_load,
        "p_load": p_load,
    }


def save_phase_plots(model_name, phase, dates, a_ssc, p_ssc, a_load, p_load, metrics, output_dir):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(dates, a_ssc, s=14, alpha=0.75, color="#111111", edgecolors="none", label="Observed SSC")
    ax.plot(dates, p_ssc, color="#0f766e", lw=2.0, label=f"{model_name} Predicted SSC")
    ax.set_title(
        f"{model_name} - {phase.capitalize()} SSC: R2={metrics['R2_SSC']:.3f} | NSE={metrics['NSE_SSC']:.3f} | "
        f"RMSE={metrics['RMSE_SSC']:.2f} | PBIAS={metrics['PBIAS_SSC_%']:.2f}%"
    )
    ax.set_ylabel("SSC (mg/L)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(model_dir / f"{phase.capitalize()}_SSC_Hydrograph.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(dates, a_load, s=14, alpha=0.75, color="#1d4ed8", edgecolors="none", label="Observed Load")
    ax.plot(dates, p_load, color="#166534", lw=2.0, label=f"{model_name} Predicted Load")
    ax.set_title(
        f"{model_name} - {phase.capitalize()} Load: R2={metrics['R2_Load']:.3f} | NSE={metrics['NSE_Load']:.3f} | "
        f"RMSE={metrics['RMSE_Load']:.2f} | PBIAS={metrics['PBIAS_Load_%']:.2f}%"
    )
    ax.set_ylabel("Sediment Load (tons/day)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(model_dir / f"{phase.capitalize()}_Load_Hydrograph.png", dpi=150)
    plt.close(fig)


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


class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def make_ann_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_lstm_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_torch_model(model, loaders, y_scaler, epochs):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val_r2 = -np.inf
    wait = 0

    for _ in range(epochs):
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
                pred = model(xb.to(DEVICE)).cpu().numpy().ravel()
                val_preds.append(pred)
                val_true.append(yb.numpy().ravel())

        pred_val = inverse_scale(np.concatenate(val_preds), y_scaler)
        true_val = inverse_scale(np.concatenate(val_true), y_scaler)
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


def evaluate_flat_model(model_name, model, data, output_dir, rows):
    for phase in ["train", "val", "test"]:
        preds_scaled = model.predict(data["X"][phase])
        p_ssc = inverse_scale(preds_scaled, data["y_scaler"])
        a_ssc = inverse_scale(data["y"][phase], data["y_scaler"])
        q_vals = data["q_m3s"][phase]

        metrics = compute_metrics(a_ssc, p_ssc, q_vals)
        rows.append({
            "Model": model_name,
            "Phase": phase,
            "R2_SSC": round(metrics["R2_SSC"], 4),
            "RMSE_SSC": round(metrics["RMSE_SSC"], 2),
            "NSE_SSC": round(metrics["NSE_SSC"], 4),
            "PBIAS_SSC_%": round(metrics["PBIAS_SSC_%"], 2),
            "Mass_Err_SSC_%": round(metrics["Mass_Err_SSC_%"], 2),
            "R2_Load": round(metrics["R2_Load"], 4),
            "RMSE_Load": round(metrics["RMSE_Load"], 2),
            "NSE_Load": round(metrics["NSE_Load"], 4),
            "PBIAS_Load_%": round(metrics["PBIAS_Load_%"], 2),
            "Mass_Err_Load_%": round(metrics["Mass_Err_Load_%"], 2),
        })
        save_phase_plots(
            model_name,
            phase,
            data["dates"][phase],
            a_ssc,
            p_ssc,
            metrics["a_load"],
            metrics["p_load"],
            metrics,
            output_dir,
        )


def evaluate_torch_model(model_name, model, data, output_dir, rows):
    model.eval()
    for phase in ["train", "val", "test"]:
        x = torch.tensor(data["X"][phase], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            preds_scaled = model(x).cpu().numpy().ravel()

        p_ssc = inverse_scale(preds_scaled, data["y_scaler"])
        a_ssc = inverse_scale(data["y"][phase], data["y_scaler"])
        q_vals = data["q_m3s"][phase]

        metrics = compute_metrics(a_ssc, p_ssc, q_vals)
        rows.append({
            "Model": model_name,
            "Phase": phase,
            "R2_SSC": round(metrics["R2_SSC"], 4),
            "RMSE_SSC": round(metrics["RMSE_SSC"], 2),
            "NSE_SSC": round(metrics["NSE_SSC"], 4),
            "PBIAS_SSC_%": round(metrics["PBIAS_SSC_%"], 2),
            "Mass_Err_SSC_%": round(metrics["Mass_Err_SSC_%"], 2),
            "R2_Load": round(metrics["R2_Load"], 4),
            "RMSE_Load": round(metrics["RMSE_Load"], 2),
            "NSE_Load": round(metrics["NSE_Load"], 4),
            "PBIAS_Load_%": round(metrics["PBIAS_Load_%"], 2),
            "Mass_Err_Load_%": round(metrics["Mass_Err_Load_%"], 2),
        })
        save_phase_plots(
            model_name,
            phase,
            data["dates"][phase],
            a_ssc,
            p_ssc,
            metrics["a_load"],
            metrics["p_load"],
            metrics,
            output_dir,
        )


def main():
    data_path = _locate_data_file("data.xlsx")
    project_root = _locate_project_root()
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = project_root / "outputs" / f"Unified_AllModels_QRainLag_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Unified all-model script. Output: {output_dir}")
    print(f"[PATH] Data: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    feature_cols = build_feature_list()
    print(f"[RUN] Feature set ({len(feature_cols)}): {feature_cols}")

    proc = DataProcessor(data_path)
    split_dfs = proc.split()

    flat_data = prepare_flat_data(split_dfs, feature_cols)
    lstm_data = prepare_lstm_data(split_dfs, feature_cols, seq_len=SEQ_LEN)

    rows = []

    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVR": SVR(kernel="rbf", C=8.0, epsilon=0.05, gamma="scale"),
    }

    for name, model in models.items():
        print(f"[TRAIN] {name}")
        model.fit(flat_data["X"]["train"], flat_data["y"]["train"])
        evaluate_flat_model(name, model, flat_data, output_dir, rows)

    print("[TRAIN] ANN")
    ann = ANNRegressor(in_dim=len(feature_cols))
    ann_loaders = {
        "train": make_ann_loader(flat_data["X"]["train"], flat_data["y"]["train"], shuffle=True),
        "val": make_ann_loader(flat_data["X"]["val"], flat_data["y"]["val"], shuffle=False),
    }
    ann, ann_val_r2 = train_torch_model(ann, ann_loaders, flat_data["y_scaler"], ANN_EPOCHS)
    print(f"        -> Best Val R2: {ann_val_r2:.4f}")
    evaluate_torch_model("ANN", ann, flat_data, output_dir, rows)

    print("[TRAIN] LSTM")
    lstm = LSTMRegressor(in_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.1)
    lstm_loaders = {
        "train": make_lstm_loader(lstm_data["X"]["train"], lstm_data["y"]["train"], shuffle=True),
        "val": make_lstm_loader(lstm_data["X"]["val"], lstm_data["y"]["val"], shuffle=False),
    }
    lstm, lstm_val_r2 = train_torch_model(lstm, lstm_loaders, lstm_data["y_scaler"], LSTM_EPOCHS)
    print(f"        -> Best Val R2: {lstm_val_r2:.4f}")
    evaluate_torch_model("LSTM", lstm, lstm_data, output_dir, rows)

    rep_df = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print(rep_df.to_string(index=False))
    print("=" * 120)

    rep_df.to_csv(output_dir / "Unified_AllModels_Metrics.csv", index=False)
    print(f"[DONE] Saved metrics + plots in: {output_dir}")


if __name__ == "__main__":
    main()
