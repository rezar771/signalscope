"""Calendar-based pairing without interpolation or input mutation."""

import pandas as pd

from signalscope.sources.base import TimeSeries


def lagged_pairs(driver: pd.Series, target: pd.Series, lag: int = 0) -> pd.DataFrame:
    """Pair target[t] with driver[t - lag]; a positive lag means the driver leads."""
    pairs = pd.concat([driver.shift(lag, freq="D").rename("driver"), target.rename("target")],
                      axis=1, join="inner").dropna().sort_index()
    pairs.index.name = "date"
    return pairs


def align(driver: TimeSeries, target: TimeSeries) -> pd.DataFrame:
    """Return the observed common dates at lag zero."""
    return lagged_pairs(driver.history, target.history)


def has_gaps(df: pd.DataFrame) -> bool:
    """Whether the index omits days between its first and last observation."""
    return len(df) > 1 and not df.index.equals(pd.date_range(df.index.min(), df.index.max()))
