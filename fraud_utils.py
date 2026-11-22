"""Helper functions untuk Kasus A: deteksi penipuan pembukaan rekening (BAF Base)."""
import numpy as np
import pandas as pd

TARGET = "fraud_bool"
CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
# Kolom yang memakai -1 sebagai penanda "tidak ada data" (sesuai datasheet BAF)
MISSING_NEG1 = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
# Kolom konstan / tidak informatif di varian Base
DROP_COLS = ["device_fraud_count", "month"]


def load_months(path, months=(0, 1, 2, 3), chunksize=200_000):
    """Baca CSV per potongan dan simpan hanya baris dengan month di `months`."""
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        parts.append(chunk[chunk["month"].isin(months)])
    return pd.concat(parts, ignore_index=True)


def prepare(df, categories=None):
    """Ubah dataframe mentah menjadi fitur siap model (tanpa label).

    categories: dict kolom -> list kategori (dari data latih) agar dtype
    konsisten saat inferensi. Jika None, dihitung dari df.
    """
    X = df.drop(columns=[c for c in [TARGET] + DROP_COLS if c in df.columns]).copy()
    for c in MISSING_NEG1:
        X[c + "_missing"] = (X[c] == -1).astype("int8")
        X.loc[X[c] == -1, c] = np.nan
    if categories is None:
        categories = {c: sorted(X[c].astype(str).unique()) for c in CAT_COLS}
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X, categories


def tpr_at_fpr(y_true, score, fpr_target=0.05):
    """Return (TPR, threshold, FPR sebenarnya) pada FPR <= fpr_target."""
    y_true = np.asarray(y_true)
    neg = np.sort(score[y_true == 0])
    thr = np.quantile(neg, 1 - fpr_target)  # skor ke-95 dari nasabah jujur
    pred = score > thr
    tpr = pred[y_true == 1].mean()
    fpr = pred[y_true == 0].mean()
    return float(tpr), float(thr), float(fpr)
