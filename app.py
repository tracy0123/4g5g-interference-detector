# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from utils import fast_load, align_smooth, feat_4g, get_enhanced_features

st.set_page_config(page_title="4G/5G干扰检测", layout="wide")

# ---------- 加载模型（只加载一次） ----------
@st.cache_resource
def load_models():
    rf_4g=joblib.load("models/rf_4g.pkl")
    scaler_4g=joblib.load("models/scaler_4g.pkl")
    lgb_5g=joblib.load("models/lgb_5g.pkl")
    scaler_5g=joblib.load("models/scaler_5g.pkl")
    return rf_4g, scaler_4g, lgb_5g, scaler_5g

rf_4g, scaler_4g, lgb_5g, scaler_5g=load_models()

# ---------- 页面标题 ----------
st.title("📡 4G/5G PRB干扰检测（网页验证版）")
mode=st.radio("选择网络类型", ["4G", "5G"])  # 已去掉括号说明

# ---------- 上传Excel ----------
uploaded=st.file_uploader("上传PRB数据Excel（每行一个样本）", type="xlsx")

if uploaded:
    st.sidebar.subheader("📊 数据预览")

    # ---------- 进度条 + 百分比提示 ----------
    progress_bar=st.progress(0)
    status_text=st.empty()

    # 1. 加载数据
    status_text.markdown("**[1/4] 正在加载数据...**")
    progress_bar.progress(10)
    X=fast_load(uploaded)
    st.sidebar.write(f"样本数：{X.shape[0]}，PRB长度：{X.shape[1]}")

    # 2. 对齐平滑
    status_text.markdown("**[2/4] 正在对齐与平滑...**")
    progress_bar.progress(30)

    X_align,=align_smooth([X])

    # 3. 特征提取 + 预测
    status_text.markdown("**[3/4] 特征计算与模型预测...**")
    progress_bar.progress(60)

    if mode == "4G":
        F=feat_4g(X_align)
        F_scaled=scaler_4g.transform(F)
        pred=rf_4g.predict(F_scaled)
        prob=rf_4g.predict_proba(F_scaled)[:, 1]
    else:
        F=get_enhanced_features(X_align)
        F_scaled=scaler_5g.transform(F)
        pred=lgb_5g.predict(F_scaled)
        prob=lgb_5g.predict_proba(F_scaled)[:, 1]

    # 4. 排序准备绘图
    status_text.markdown("**[4/4] 结果排序与绘图准备...**")
    progress_bar.progress(90)

    res_df=pd.DataFrame({
        "原始行号": list(range(len(pred))),
        "是否干扰（1=是）": pred,
        "干扰概率": np.round(prob, 3)
    })
    res_df_sorted=res_df.sort_values("干扰概率", ascending=False).reset_index(drop=True)

    progress_bar.progress(100)
    status_text.success("✅ 处理完成！")

    # 隐藏进度条（清爽）
    progress_bar.empty()
    status_text.empty()

    # ---------- 结果展示 ----------
    st.subheader("✅ 检测结果（按干扰概率降序）")
    st.dataframe(res_df_sorted, use_container_width=True)

    # ---------- 只画概率最高的前3个样本 ----------
    st.subheader("📈 干扰概率最高的前3个样本曲线")
    top3_indices=res_df_sorted["原始行号"].head(3).tolist()

    for idx in top3_indices:
        fig, ax=plt.subplots(figsize=(10, 3))
        ax.plot(X_align[idx], color="#d62728", linewidth=2)
        ax.set_title(f"样本 {idx} | 干扰={pred[idx]} | 干扰概率={prob[idx]:.3f}")
        ax.set_xlabel("PRB索引")
        ax.set_ylabel("信号值 (dBm)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)