"""Walk-forward optimization windows.

Current implementation: returns a single fixed 70/30 train/test split
regardless of the window-length and mode arguments (they are accepted and
validated so the sidebar wiring is final, but not yet applied).

Planned enhancement: a proper rolling/expanding window generator, with
successive in-sample windows of ``in_sample_len`` periods (anchored at the
start when ``mode == "expanding"``), each followed by an ``out_sample_len``
out-of-sample window, with strategy parameters re-fit per window and
in-sample vs. out-of-sample performance reported separately.
"""

import pandas as pd


def split_windows(
    data: pd.DataFrame,
    in_sample_len: int,
    out_sample_len: int,
    mode: str,
) -> list:
    """Generate (train, test) windows over ``data``'s DatetimeIndex.

    Parameters
    ----------
    data : pd.DataFrame
        Price panel; only its index is used.
    in_sample_len, out_sample_len : int
        Window lengths in trading days (not yet applied).
    mode : str
        ``"rolling"`` or ``"expanding"`` (not yet applied).

    Returns
    -------
    list[dict]
        One dict per window: ``{"window": int, "train": (start, end),
        "test": (start, end)}`` with pandas Timestamps. The current
        implementation always returns exactly one window covering the full
        sample.
    """
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")

    idx = data.index
    if len(idx) < 10:
        return []

    split_at = int(len(idx) * 0.7)
    return [
        {
            "window": 1,
            "train": (idx[0], idx[split_at - 1]),
            "test": (idx[split_at], idx[-1]),
        }
    ]
