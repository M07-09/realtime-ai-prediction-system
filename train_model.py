"""
Train the LSTM forecaster - one command, start to finish.

This single script does everything the model needs:

    1. download 20,000 real 1-minute BTC/USDT candles from Binance (cached)
    2. build the 7 features and the target
    3. split chronologically 70 / 15 / 15  (the test set is the most recent data)
    4. fit the scaler on the training split only
    5. train the LSTM with early stopping
    6. evaluate with MAE, MSE, RMSE, MAPE and R2 plus a naive baseline
    7. save the model, the scaler, the metrics and 5 plots into models/

What the model predicts
-----------------------
Given the last 60 one-minute candles, it predicts the log return of the NEXT
candle, i.e. the price 60 seconds into the future. The price is reconstructed as

        predicted_price = last_close * exp(predicted_log_return)

Usage
-----
    python train_model.py                      # download if needed, then train
    python train_model.py --refresh            # force a fresh download
    python train_model.py --epochs 150
    python train_model.py --bars 40000 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    HISTORY_BARS,
    HISTORY_CSV,
    INTERVAL,
    INTERVAL_SECONDS,
    LEARNING_RATE,
    MAX_EPOCHS,
    METRICS_PATH,
    MODELS_DIR,
    MODEL_PATH,
    PLOT_DIR,
    RANDOM_SEED,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    SYMBOL,
    TARGET_COLUMN,
    TRAIN_HISTORY_PATH,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from app.data import StandardScaler, build_features, chronological_split, make_sequences
from app.model import PriceLSTM, pick_device, save_model
from app.utils import directional_accuracy, get_logger, regression_report, save_json

log = get_logger("train_model", "training.log")


# ==========================================================================
# 1. Data
# ==========================================================================
def load_history(refresh: bool, bars: int) -> pd.DataFrame:
    """Read the cached CSV, or download it from the external API."""
    if refresh or not HISTORY_CSV.exists():
        from app.crypto_api import ExternalAPIError, get_client

        log.info("Downloading %d %s candles of %s from Binance ...", bars, INTERVAL, SYMBOL)
        try:
            history = get_client().fetch_history(total_bars=bars, interval=INTERVAL)
        except ExternalAPIError as exc:
            raise SystemExit(f"Download failed: {exc}\nCheck your internet connection.") from exc
        history.to_csv(HISTORY_CSV, index=False)
        log.info("Saved %d candles to %s", len(history), HISTORY_CSV)
    else:
        log.info("Using the cached history in %s (use --refresh to re-download)", HISTORY_CSV)
        history = pd.read_csv(HISTORY_CSV, parse_dates=["open_time", "close_time"])

    if len(history) < SEQUENCE_LENGTH * 10:
        raise SystemExit(f"Not enough history ({len(history)} candles). Run with --refresh.")
    return history


def prepare_tensors(features: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """Split chronologically, fit the scaler on train only, build the sequences."""
    X_raw = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_raw = features[TARGET_COLUMN].to_numpy(dtype=np.float64)
    close = features["close"].to_numpy(dtype=np.float64)

    tr, va, te = chronological_split(len(features), TRAIN_SPLIT, VAL_SPLIT)
    log.info("Chronological split -> train %d | val %d | test %d",
             tr.stop - tr.start, va.stop - va.start, te.stop - te.start)

    # Fitted on the training rows only: no information from the future leaks in.
    scaler = StandardScaler().fit(X_raw[tr], FEATURE_COLUMNS, y_raw[tr])
    X_scaled = scaler.transform(X_raw)
    y_scaled = scaler.transform_target(y_raw)

    bundle: Dict[str, np.ndarray] = {}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        # Sequences are built inside each split, so no window ever spans a boundary.
        X_seq, y_seq, idx = make_sequences(X_scaled[sl], y_scaled[sl], SEQUENCE_LENGTH)
        global_idx = idx + sl.start
        bundle[f"X_{name}"] = X_seq
        bundle[f"y_{name}"] = y_seq
        bundle[f"close_{name}"] = close[global_idx]
        bundle[f"true_return_{name}"] = y_raw[global_idx]
        bundle[f"true_next_close_{name}"] = close[global_idx] * np.exp(y_raw[global_idx])
        log.info("  %-5s sequences: %s", name, X_seq.shape)

    return bundle, scaler


# ==========================================================================
# 2. Training
# ==========================================================================
def train(bundle: Dict[str, np.ndarray], epochs: int, device: torch.device):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_ds = TensorDataset(torch.from_numpy(bundle["X_train"]),
                             torch.from_numpy(bundle["y_train"]))
    val_ds = TensorDataset(torch.from_numpy(bundle["X_val"]),
                           torch.from_numpy(bundle["y_val"]))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PriceLSTM(input_size=len(FEATURE_COLUMNS)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=4)
    criterion = nn.MSELoss()

    history: Dict[str, object] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_state = float("inf"), None
    patience_left = EARLY_STOPPING_PATIENCE
    started = time.time()

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Training on %s | %d trainable parameters", device, n_params)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = running / len(val_ds)

        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        marker = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOPPING_PATIENCE
            marker = "  <-- best"
        else:
            patience_left -= 1

        log.info("Epoch %3d/%d | train %.6f | val %.6f | lr %.2e%s",
                 epoch, epochs, train_loss, val_loss, current_lr, marker)

        if patience_left <= 0:
            log.info("Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history.update(training_seconds=time.time() - started,
                   epochs_run=len(history["train_loss"]),
                   best_val_loss=best_val,
                   parameters=n_params)
    log.info("Training finished in %.1f s (best validation loss %.6f)",
             history["training_seconds"], best_val)
    return model, history


# ==========================================================================
# 3. Evaluation
# ==========================================================================
@torch.no_grad()
def predict_scaled(model: PriceLSTM, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(X), 512):
        chunks.append(model(torch.from_numpy(X[start:start + 512]).to(device)).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.array([])


def evaluate(model, bundle, scaler, device, split: str = "test"):
    """Metrics in US-dollar price space, plus direction and a naive baseline."""
    pred_return = scaler.inverse_target(
        predict_scaled(model, bundle[f"X_{split}"], device).astype(np.float64))
    true_return = bundle[f"true_return_{split}"]
    last_close = bundle[f"close_{split}"]

    pred_price = last_close * np.exp(pred_return)
    true_price = bundle[f"true_next_close_{split}"]
    naive_price = last_close                       # "the price will not change"

    report = regression_report(true_price, pred_price)
    baseline = regression_report(true_price, naive_price)

    result = {
        "split": split,
        "samples": int(len(true_price)),
        "price_space": report,
        "naive_baseline_price_space": baseline,
        "return_space": regression_report(true_return, pred_return),
        "directional_accuracy_pct": directional_accuracy(true_return, pred_return),
        "improvement_vs_naive_rmse_pct": float(
            (baseline["RMSE"] - report["RMSE"]) / baseline["RMSE"] * 100.0
        ) if baseline["RMSE"] else float("nan"),
    }

    log.info("--- %s metrics (price space, USDT) ---", split.upper())
    for key in ("MAE", "MSE", "RMSE", "MAPE", "R2"):
        log.info("  %-5s : %12.6f", key, report[key])
    log.info("  Directional accuracy : %.2f %%", result["directional_accuracy_pct"])
    log.info("  Naive baseline RMSE  : %.4f  (model is %.2f%% better)",
             baseline["RMSE"], result["improvement_vs_naive_rmse_pct"])

    arrays = {"true_price": true_price, "pred_price": pred_price,
              "true_return": true_return, "pred_return": pred_return}
    return result, arrays


# ==========================================================================
# 4. Plots
# ==========================================================================
def make_plots(features: pd.DataFrame, history: Dict, arrays: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.autolayout": True})

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(features["open_time"], features["close"], lw=0.8, color="#1f77b4")
    ax.set_title(f"{SYMBOL} - historical close price ({INTERVAL} candles, {len(features):,} bars)")
    ax.set_xlabel("UTC time"); ax.set_ylabel("Price (USDT)")
    fig.savefig(PLOT_DIR / "1_historical_prices.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_title("LSTM training curve (MSE on standardised log returns)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()
    fig.savefig(PLOT_DIR / "2_loss_curves.png"); plt.close(fig)

    n_show = min(400, len(arrays["true_price"]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(arrays["true_price"][-n_show:], label="actual price", lw=1.4, color="#111111")
    ax.plot(arrays["pred_price"][-n_show:], label="LSTM prediction", lw=1.1,
            color="#d62728", alpha=0.85)
    ax.set_title(f"Actual vs predicted next-minute price - last {n_show} test candles")
    ax.set_xlabel("test candle"); ax.set_ylabel("Price (USDT)"); ax.legend()
    fig.savefig(PLOT_DIR / "3_actual_vs_predicted.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(arrays["true_price"], arrays["pred_price"], s=4, alpha=0.35, color="#2ca02c")
    lo = float(min(arrays["true_price"].min(), arrays["pred_price"].min()))
    hi = float(max(arrays["true_price"].max(), arrays["pred_price"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1)
    ax.set_title("Predicted vs actual price (test split)")
    ax.set_xlabel("actual (USDT)"); ax.set_ylabel("predicted (USDT)")
    fig.savefig(PLOT_DIR / "4_prediction_scatter.png"); plt.close(fig)

    errors = arrays["pred_price"] - arrays["true_price"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=60, color="#9467bd", alpha=0.85)
    ax.axvline(0, color="#d62728", ls="--", lw=1)
    ax.set_title(f"Prediction error distribution "
                 f"(mean {errors.mean():.2f}, std {errors.std():.2f} USDT)")
    ax.set_xlabel("predicted - actual (USDT)"); ax.set_ylabel("count")
    fig.savefig(PLOT_DIR / "5_error_distribution.png"); plt.close(fig)

    log.info("Saved 5 plots to %s", PLOT_DIR)


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Train the LSTM price forecaster")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--bars", type=int, default=HISTORY_BARS)
    parser.add_argument("--refresh", action="store_true", help="re-download the history")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    history_df = load_history(args.refresh, args.bars)
    log.info("History: %d candles, %s -> %s",
             len(history_df), history_df["open_time"].iloc[0], history_df["open_time"].iloc[-1])

    features = build_features(history_df, with_target=True)
    log.info("Feature matrix: %d rows x %d features", len(features), len(FEATURE_COLUMNS))

    bundle, scaler = prepare_tensors(features)
    device = pick_device(args.device)
    model, train_history = train(bundle, args.epochs, device)

    test_metrics, arrays = evaluate(model, bundle, scaler, device, "test")
    val_metrics, _ = evaluate(model, bundle, scaler, device, "val")

    scaler.save(SCALER_PATH)
    save_model(model, MODEL_PATH, extra={
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon_bars": FORECAST_HORIZON,
        "forecast_horizon_seconds": INTERVAL_SECONDS * FORECAST_HORIZON,
        "features": FEATURE_COLUMNS,
    })
    save_json(METRICS_PATH, {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon_seconds": INTERVAL_SECONDS * FORECAST_HORIZON,
        "features": FEATURE_COLUMNS,
        "target": "next-candle log return, reconstructed to a price",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "data": {
            "total_bars": int(len(history_df)),
            "usable_rows": int(len(features)),
            "period_start": str(history_df["open_time"].iloc[0]),
            "period_end": str(history_df["open_time"].iloc[-1]),
            "train_sequences": int(len(bundle["X_train"])),
            "val_sequences": int(len(bundle["X_val"])),
            "test_sequences": int(len(bundle["X_test"])),
        },
        "training": {
            "device": str(device),
            "epochs_run": train_history["epochs_run"],
            "training_seconds": train_history["training_seconds"],
            "best_val_loss": train_history["best_val_loss"],
            "parameters": train_history["parameters"],
        },
        "test": test_metrics,
        "validation": val_metrics,
    })
    save_json(TRAIN_HISTORY_PATH, dict(train_history))

    make_plots(features, train_history, arrays)

    price = test_metrics["price_space"]
    print("\n" + "=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Test MAE   : {price['MAE']:>12,.2f} USDT")
    print(f"  Test RMSE  : {price['RMSE']:>12,.2f} USDT")
    print(f"  Test MAPE  : {price['MAPE']:>12.4f} %")
    print(f"  Test R2    : {price['R2']:>12.4f}")
    print(f"\n  Saved to   : {MODELS_DIR}")
    print("  Next step  : python -m app.main   then   python -m streamlit run streamlit_app.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
