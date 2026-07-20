"""Small CSV-backed annotation store shared by the Streamlit apps."""

import os
from pathlib import Path

import pandas as pd


def read_csv_if_exists(path):
    """Return an empty frame when an annotation CSV has not been created yet."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def upsert_row(path, row, key_column, existing=None, columns=None):
    """Insert or replace the last row matching ``key_column``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(row)
    if existing is None:
        existing = read_csv_if_exists(path)
    if existing.empty:
        existing = pd.DataFrame(columns=columns)

    new_row = pd.DataFrame([row], columns=columns)
    if key_column not in existing.columns:
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        matches = existing[key_column].astype(str) == str(row[key_column])
        if not matches.any():
            updated = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = existing.copy()
            row_index = matches[matches].index[-1]
            for column in columns:
                updated.loc[row_index, column] = new_row.iloc[0][column]

    updated.to_csv(path, index=False)
