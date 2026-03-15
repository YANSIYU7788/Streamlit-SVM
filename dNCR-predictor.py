import streamlit as st
import pandas as pd
import joblib
import shap
import os

# =========================
# 1. 加载模型、标准化器和特征列
# =========================
# 使用相对路径，这样上传到 GitHub 后 Streamlit Cloud 能找到文件
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
categorical_cols = [col for col in feature_cols if col not in continuous_cols]

# 生成输入框
input_data = {}
for col in feature_cols:
    if col in continuous_cols:
        input_data[col] = st.number_input(f"{col}:", value=0)
    else:
        # Education 完整 0-4，其他分类特征默认 0/1
        if col == "Education":
            input_data[col] = st.selectbox(f"{col}:", options=list(range(0, 5)))
        else:
            input_data[col] = st.selectbox(f"{col}:", options=[0, 1])

X_input = pd.DataFrame([input_data])

# 标准化连续变量
X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])

# =========================
# 3. 预测按钮
# =========================
if st.button("预测"):
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"预测概率: {pred_prob[0]:.4f}")
    st.write(f"预测结果: {'yes' if pred_label[0] == 1 else 'no'}")

    # =========================
    # 4. SHAP 值和力图
    # =========================
    # 生成背景数据集（可用均值或中位数）
    background_data = pd.DataFrame([{
        col: 0 if col in continuous_cols else 1 for col in feature_cols
    }])

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    # 如果返回多维，取第一个样本
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    st.write("各特征的 SHAP 值：")
    for name, val in zip(feature_cols, shap_values):
        st.write(f"{name}: {val:.4f}")

    # 生成力图并保存 HTML
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
