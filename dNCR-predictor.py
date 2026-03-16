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
st.title("SVM predict dNCR")
st.write("Please enter patient characteristics：")

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
if st.button("predict"):
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Preducted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # 创建一个合理的背景数据集（用特征的均值/中位数）
    background_data = pd.DataFrame([{
        'Age': 0,
        'Education': 1,
        'MOCA_Score': 0,
        'Operation_Time': 0,
        'GFR': 0,
        'Weakened_1': 0,
        'Depression_1': 0,
        'Nutritional_Risk_1': 0
    }])

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    # 确保是一维数组
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # 打印 SHAP 值查看
    st.write("SHAP values of each feature：")
    for name, val in zip(feature_cols, shap_values):
        st.write(f"{name}: {val:.4f}")

    # 生成力图
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=X_input.iloc[0].values,
        feature_names=feature_cols,
        matplotlib=False
    )

    shap.save_html("shap_force_plot.html", force_plot)

    st.subheader("SHAP force plot of the prediction")
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=400)

