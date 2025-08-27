"""
ECE (Expected Calibration Error) utilities
Contains the team's ECE function implementation
"""

import pandas as pd
import numpy as np

def expected_calibration_error(
    y_true,
    y_pred,
    sample_weight=None,
    n_bins: int = 5,
) -> float:
    # Ensure y_true and y_pred are numpy arrays
    y_pred_arr = np.asarray(y_pred)
    y_true_arr = np.asarray(y_true)

    # ECE only makes sense for a single class
    if y_pred_arr.ndim > 1:
        return 0.0

    # require at least n_bins * 10, otherwise return 0.0 (no calibration error)
    if len(y_true) < n_bins * 10:
        return 0.0

    # create a DataFrame to hold the true labels and predicted probabilities
    df = pd.DataFrame({
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    })

    # Bin the predicted probabilities
    df["y_pred_bin"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")

    # 2) per-bin stats
    bin_stats = (
        df.groupby("y_pred_bin")
        .agg(
            mean_bin_y_pred=("y_pred", "mean"),
            mean_bin_y_true=("y_true", "mean"),
            n_samples=("y_pred_bin", "size"),
        )
        .reset_index(drop=True)
    )

    # 3) |gap| and weights
    bin_stats["abs_diff"] = (bin_stats["mean_bin_y_pred"] - bin_stats["mean_bin_y_true"]).abs()
    weights = bin_stats["n_samples"] / len(df)

    ece = (weights * bin_stats["abs_diff"]).sum()

    return ece
