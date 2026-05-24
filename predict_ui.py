"""
predict_ui.py — California Housing Price Predictor
====================================================
Usage:
    python predict_ui.py              # interactive mode (prompts for inputs)
    python predict_ui.py --demo       # run on 3 preset sample houses
    python predict_ui.py --help       # show this help

Requires:
    saved_model/linear_regression_model.pkl
    saved_model/scaler.pkl
    (both created by task1_ml_linear_regression.ipynb)

Author : Sai Sohan | ACE Engineering College | AI & ML
"""

import pickle
import argparse
import sys
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "saved_model", "linear_regression_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "scaler.pkl")

# ── Feature definitions ────────────────────────────────────────────────────────
FEATURES = [
    ("MedInc",     "Median Income of block (in $10k units, e.g. 5.0 = $50,000)",  0.5, 15.0,  3.87),
    ("HouseAge",   "Median House Age in the block (years)",                        1.0, 52.0,  28.7),
    ("AveRooms",   "Average number of rooms per household",                        1.0, 30.0,   5.4),
    ("AveBedrms",  "Average number of bedrooms per household",                     0.5, 10.0,   1.1),
    ("Population", "Block population",                                            10.0, 5000.,  1425),
    ("AveOccup",   "Average occupants per household",                              1.0, 10.0,   3.0),
    ("Latitude",   "Block latitude (California: ~32.5 to 42.0)",                  32.5, 42.0,  35.6),
    ("Longitude",  "Block longitude (California: ~-124.4 to -114.3)",           -124.4,-114.3,-119.6),
]

DEMO_HOUSES = [
    {
        "label":      "Affluent Coastal (e.g. Malibu-style)",
        "MedInc":     10.0,
        "HouseAge":   20.0,
        "AveRooms":   8.0,
        "AveBedrms":  1.5,
        "Population": 600.0,
        "AveOccup":   2.5,
        "Latitude":   34.0,
        "Longitude": -118.5,
    },
    {
        "label":      "Mid-range Suburban (e.g. San Jose suburb)",
        "MedInc":     5.0,
        "HouseAge":   30.0,
        "AveRooms":   5.5,
        "AveBedrms":  1.1,
        "Population": 1200.0,
        "AveOccup":   3.0,
        "Latitude":   37.3,
        "Longitude": -121.9,
    },
    {
        "label":      "Budget Inland (e.g. Central Valley)",
        "MedInc":     2.0,
        "HouseAge":   40.0,
        "AveRooms":   4.0,
        "AveBedrms":  1.2,
        "Population": 2500.0,
        "AveOccup":   3.8,
        "Latitude":   36.7,
        "Longitude": -119.8,
    },
]


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    """Load saved model and scaler; exit with clear error if missing."""
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"\n❌  Model file not found: {MODEL_PATH}\n"
            "    Please run task1_ml_linear_regression.ipynb first to generate the model.\n"
        )
    if not os.path.exists(SCALER_PATH):
        sys.exit(
            f"\n❌  Scaler file not found: {SCALER_PATH}\n"
            "    Please run task1_ml_linear_regression.ipynb first.\n"
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(model, scaler, house: dict) -> float:
    """Return prediction in $100k units."""
    import numpy as np
    feature_names = [f[0] for f in FEATURES]
    X = [[house[k] for k in feature_names]]
    X_sc = scaler.transform(X)
    return float(model.predict(X_sc)[0])


# ── Display helpers ────────────────────────────────────────────────────────────
def banner():
    print("\n" + "=" * 60)
    print("  🏠  California Housing Price Predictor")
    print("      Linear Regression Model  |  ACE Engineering College")
    print("=" * 60)


def display_result(label: str, house: dict, pred_100k: float):
    print(f"\n  {'─' * 50}")
    if label:
        print(f"  🏘  {label}")
        print(f"  {'─' * 50}")
    feature_names = [f[0] for f in FEATURES]
    for k in feature_names:
        print(f"    {k:<14}: {house[k]}")
    print(f"  {'─' * 50}")
    dollars = pred_100k * 100_000
    print(f"  💰  Predicted Median House Value")
    print(f"      ➜  ${dollars:,.0f}  ({pred_100k:.3f} × $100k units)")
    # Simple confidence band from RMSE
    rmse = 0.6582
    low  = max(0, (pred_100k - rmse)) * 100_000
    high = (pred_100k + rmse) * 100_000
    print(f"      ~68% confidence range: ${low:,.0f} – ${high:,.0f}")
    print(f"  {'─' * 50}")


# ── Interactive mode ───────────────────────────────────────────────────────────
def interactive_mode(model, scaler):
    banner()
    print("\n  Enter block-level statistics to predict the median house value.")
    print("  Press Enter to accept the default (average) value shown in [ ].\n")

    house = {}
    for name, description, lo, hi, default in FEATURES:
        while True:
            prompt = f"  {name} ({lo}–{hi}) [{default}]: "
            raw = input(prompt).strip()
            if raw == "":
                house[name] = default
                break
            try:
                val = float(raw)
                if not (lo <= val <= hi):
                    print(f"    ⚠  Please enter a value between {lo} and {hi}.")
                    continue
                house[name] = val
                break
            except ValueError:
                print("    ⚠  Please enter a numeric value.")

    pred = predict(model, scaler, house)
    display_result("Your Input", house, pred)

    # Ask to predict another
    again = input("\n  Predict another house? [y/N]: ").strip().lower()
    if again == "y":
        interactive_mode(model, scaler)
    else:
        print("\n  Thank you for using the California Housing Predictor! 👋\n")


# ── Demo mode ──────────────────────────────────────────────────────────────────
def demo_mode(model, scaler):
    banner()
    print("\n  Running predictions on 3 preset example houses...\n")
    for h in DEMO_HOUSES:
        label = h.pop("label")
        pred  = predict(model, scaler, h)
        display_result(label, h, pred)
        h["label"] = label          # restore for potential re-use
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict California median house values using a saved Linear Regression model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run predictions on 3 preset sample houses instead of prompting."
    )
    args = parser.parse_args()

    model, scaler = load_model()

    if args.demo:
        demo_mode(model, scaler)
    else:
        interactive_mode(model, scaler)


if __name__ == "__main__":
    main()
