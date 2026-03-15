import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# =========================
# 1. 加载模型、标准化器和特征列（相对路径）
# =========================
model_path = "svm_final_model.pkl"
scaler_path = "svm_scaler.pkl"
feature_cols_path = "svm_feature_columns.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_cols = joblib.load(feature_cols_path)

# =========================
# 2. Streamlit 界面
# =========================
st.title("SVM 预测 dNCR")
st.write("请输入患者特征：")

continuous_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']

# 自动识别分类特征
categorical_cols = [col for col in feature_cols if col not in continuous_cols]

input_data = {}
for col in feature_cols:
    if col in continuous_cols:
        input_data[col] = st.number_input(f"{col}:", value=0)
    else:
        max_val = 4 if 'Education' in col else 1
        input_data[col] = st.selectbox(f"{col}:", options=list(range(0, max_val + 1)))

X_input = pd.DataFrame([input_data])
X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])

# =========================
# 3. 预测按钮
# =========================
if st.button("预测"):
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"预测概率: {pred_prob[0]:.4f}")
    st.write(f"预测结果: {'yes' if pred_label[0]==1 else 'no'}")

    # =========================
    # 4. SHAP解释
    # =========================
    # 背景数据（可用均值或采样）
    background_data = pd.DataFrame([X_input.iloc[0].values], columns=feature_cols)

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:,1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    # 确保 shap_values 是一维
    if isinstance(shap_values, list) or (hasattr(shap_values, 'shape') and len(shap_values.shape) > 1):
        shap_values = np.array(shap_values)[0]

    st.write("各特征的 SHAP 值：")
    for name, val in zip(feature_cols, shap_values):
        if isinstance(val, (list, np.ndarray)):
            val_to_show = float(val[0])
        else:
            val_to_show = float(val)
        st.write(f"{name}: {val_to_show:.4f}")

    # 生成力图
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=X_input.iloc[0].values,
        feature_names=feature_cols,
        matplotlib=False
    )

    shap.save_html("shap_force_plot.html", force_plot)

    st.subheader("模型预测的 SHAP 力图")
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=400)
