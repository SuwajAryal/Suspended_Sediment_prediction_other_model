#!/usr/bin/env python
"""
Traditional Machine Learning Baselines for Daily SSC and Load
==============================================================================
Description:
This script models the daily Suspended Sediment Concentration (SSC) and 
calculates the Suspended Sediment Load (SSL) using powerful, non-deep-learning 
algorithms: XGBoost, Random Forest, and Support Vector Regression (SVR).

No physical equations or regime constraints are used. The models rely on 
engineered temporal features (lags, rolling averages, seasonality) to capture 
the time-series dynamics.
"""

import warnings
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- CONFIG ---
SEED = 42
np.random.seed(SEED)

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
RUN_TAG    = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"Baseline_TradML_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- COMBINED FEATURE SET ---
BASE_FEATURES = [
    "Q_m3s", "d50_mm", "rainfall_mm", "temp_C", "VS", "Soil_Moisture", "NDVI",
    "dQ", "Q_ratio", "HI_proxy", "API", "P_MA3", "P_MA7", "SPI_30",
    "month_sin", "month_cos", "days_gap", "Q_lag1"
]
TARGET_COL = "SSC_mgL"

# --- DATA PROCESSOR (Flat 2D Arrays for Scikit-Learn/XGBoost) ---
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
            if col not in df.columns: df[col] = 0.0

        Q = df["Q_m3s"]
        df["dQ"] = Q.diff().fillna(0.0)
        q7 = Q.rolling(7, min_periods=1).mean()
        df["Q_ratio"] = (Q / q7.replace(0, np.nan)).fillna(1.0)
        df["Q_lag1"] = Q.shift(1).ffill().fillna(0.0)

        P = df["rainfall_mm"]
        p_arr = P.values
        api_vals = np.zeros(len(df))
        for k in range(1, 31):
            shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
            api_vals += (0.9 ** k) * shifted
        df["API"] = api_vals
        df["P_MA3"] = P.rolling(3, min_periods=1).mean()
        df["P_MA7"] = P.rolling(7, min_periods=1).mean()
        
        p30 = P.rolling(30, min_periods=5).sum()
        p30mu = p30.expanding(min_periods=5).mean()
        p30std = p30.expanding(min_periods=5).std().replace(0, np.nan)
        df["SPI_30"] = ((p30 - p30mu) / p30std).fillna(0.0)

        q_maxroll = Q.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (df["dQ"] / q_maxroll).fillna(0.0)

        df["month_sin"] = np.sin(2 * np.pi * df["date_dt"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date_dt"].dt.month / 12)
        df["days_gap"] = df["date_dt"].diff().dt.days.fillna(1.0)

        # Autoregressive target feature
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
        
        # Scikit-learn models expect 2D X arrays and 1D y arrays
        X_tr = x_scaler.fit_transform(train[features].values)
        X_va = x_scaler.transform(val[features].values)
        X_te = x_scaler.transform(test[features].values)
        
        y_tr = y_scaler.fit_transform(train[[TARGET_COL]].values).ravel()
        y_va = y_scaler.transform(val[[TARGET_COL]].values).ravel()
        y_te = y_scaler.transform(test[[TARGET_COL]].values).ravel()

        return {
            "X": {"train": X_tr, "val": X_va, "test": X_te},
            "y": {"train": y_tr, "val": y_va, "test": y_te},
            "y_scaler": y_scaler,
            "dates": {
                "train": train["date_dt"].values, 
                "val": val["date_dt"].values, 
                "test": test["date_dt"].values
            },
            "q_m3s": {
                "train": train["Q_m3s"].values, 
                "val": val["Q_m3s"].values, 
                "test": test["Q_m3s"].values
            }
        }

def evaluate_and_plot(model_name, preds_scaled, data, phase, results_list):
    # Inverse transform predictions and actuals
    y_scaler = data["y_scaler"]
    
    # y_scaler expects 2D array: (n_samples, 1)
    p_ssc = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    a_ssc = y_scaler.inverse_transform(data["y"][phase].reshape(-1, 1)).flatten()
    
    # Calculate Load
    q_m3s = data["q_m3s"][phase]
    p_load = p_ssc * q_m3s * 0.0864
    a_load = a_ssc * q_m3s * 0.0864

    # Calculate Metrics
    r2_ssc = r2_score(a_ssc, p_ssc)
    rmse_ssc = np.sqrt(mean_squared_error(a_ssc, p_ssc))
    sd_obs_ssc = np.std(a_ssc)
    sd_pred_ssc = np.std(p_ssc)
    mass_err_ssc = (np.sum(p_ssc) - np.sum(a_ssc)) / np.sum(a_ssc) * 100
    
    r2_load = r2_score(a_load, p_load)
    rmse_load = np.sqrt(mean_squared_error(a_load, p_load))
    sd_obs_load = np.std(a_load)
    sd_pred_load = np.std(p_load)
    mass_err_load = (np.sum(p_load) - np.sum(a_load)) / np.sum(a_load) * 100
    
    results_list.append({
        "Model": model_name,
        "Phase": phase,
        "R2_SSC": round(r2_ssc, 4), "RMSE_SSC": round(rmse_ssc, 2), "SD_Observed_SSC": round(sd_obs_ssc, 2), "SD_Predicted_SSC": round(sd_pred_ssc, 2), "Mass_Err_SSC_%": round(mass_err_ssc, 2),
        "R2_Load": round(r2_load, 4), "RMSE_Load": round(rmse_load, 2), "SD_Observed_Load": round(sd_obs_load, 2), "SD_Predicted_Load": round(sd_pred_load, 2), "Mass_Err_Load_%": round(mass_err_load, 2)
    })
    
    # Plotting
    dates = data["dates"][phase]
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    
    # SSC Plot
    plt.figure(figsize=(12, 5))
    plt.plot(dates, a_ssc, 'k-', label='Observed SSC')
    plt.plot(dates, p_ssc, 'r--', label=f'{model_name} Predicted SSC')
    plt.title(f"{model_name} - {phase.capitalize()} Phase SSC: R2={r2_ssc:.3f}")
    plt.ylabel("SSC (mg/L)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / f"{phase.capitalize()}_SSC_Hydrograph.png", dpi=150)
    plt.close()
    
    # Load Plot
    plt.figure(figsize=(12, 5))
    plt.plot(dates, a_load, 'b-', label='Observed Load (tons/day)')
    plt.plot(dates, p_load, 'g--', label=f'{model_name} Predicted Load')
    plt.title(f"{model_name} - {phase.capitalize()} Phase Load: R2={r2_load:.3f}")
    plt.ylabel("Sediment Load (tons/day)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / f"{phase.capitalize()}_Load_Hydrograph.png", dpi=150)
    plt.close()

def main():
    print(f"[INIT] Traditional ML Baselines initializing. Saving to {OUTPUT_DIR}")
    print(f"[PATH] Using data file: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}.")
    
    proc = DataProcessor(DATA_PATH)
    feats = BASE_FEATURES + ["SSC_lag1", "SSC_lag3"]
    data = proc.split_and_scale(feats)
    
    print(f"[RUN] Features used: {len(feats)}")
    
    # Define Models
    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=700,
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
        "SVR": SVR(kernel="rbf", C=8.0, epsilon=0.05, gamma="scale")
    }
    
    results_list = []
    
    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        
        # Fit model on training data
        model.fit(data["X"]["train"], data["y"]["train"])
        
        # Evaluate on all phases
        for phase in ["train", "val", "test"]:
            preds = model.predict(data["X"][phase])
            evaluate_and_plot(name, preds, data, phase, results_list)
            
    # Compile and display results
    rep_df = pd.DataFrame(results_list)
    print("\n" + "="*95)
    print(rep_df.to_string(index=False))
    print("="*95)
    
    rep_df.to_csv(OUTPUT_DIR / "Traditional_ML_Baseline_Metrics.csv", index=False)
    print(f"\n[DONE] Execution complete. Results and sub-folders saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()