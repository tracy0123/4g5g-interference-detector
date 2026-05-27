# utils.py
import numpy as np
import pandas as pd

# ---------- 1. 加载Excel ----------
def fast_load(file_path):
    df = pd.read_excel(file_path, header=None)
    df = df.replace(["NIL", "nil", "NULL", "null", "NaN", "nan"], np.nan)
    df = df.apply(lambda r: r.fillna(r.mean()) if not r.isnull().all() else r, axis=1)
    df = df.fillna(0)
    return df.values

# ---------- 2. 对齐长度（关键） ----------
def align_smooth(arr_list):
    max_len = max(arr.shape[1] for arr in arr_list)
    padded = []
    for arr in arr_list:
        if arr.shape[1] >= max_len:
            padded.append(arr[:, :max_len])
        else:
            pad_len = max_len - arr.shape[1]
            pad_vals = arr[:, -1:] @ np.ones((1, pad_len))
            padded.append(np.hstack([arr, pad_vals]))
    return padded

# ---------- 3. 4G特征（随机森林用） ----------
def feat_4g(mat):
    feat = []
    for row in mat:
        feat.append([
            np.mean(row), np.std(row), np.max(row), np.min(row),
            np.max(row)-np.min(row)
        ])
    return np.array(feat)

# ---------- 4. 5G增强特征（LightGBM用） ----------
SEG_NUM=8
WINDOW_SIZE=5

def get_enhanced_features(mat):
    mat = np.array(mat, dtype=np.float64)
    n_sample, n_len = mat.shape
    feat_list = []
    for row in mat:
        f = []
        mean_val=np.mean(row)
        std_val=np.sqrt(np.var(row)+1e-8)
        max_val=np.max(row)
        min_val=np.min(row)
        p2p=max_val-min_val
        diff=np.diff(row)
        diff_mean=np.mean(np.abs(diff))
        f.extend([mean_val, std_val, max_val, min_val, p2p, diff_mean])

        seg_len=n_len//SEG_NUM
        for i in range(SEG_NUM):
            seg=row[i*seg_len:(i+1)*seg_len]
            f.append(np.mean(seg))

        roll_std=[np.std(row[i:i+WINDOW_SIZE]) for i in range(n_len-WINDOW_SIZE+1)]
        f.append(np.mean(roll_std))

        f.append(np.mean(row)-np.median(row))
        feat_list.append(f)
    return np.array(feat_list)