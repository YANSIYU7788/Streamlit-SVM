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

# 将 Education 作为连续变量（等级变量）
continuous_cols = ['Age', 'Education', 'MOCA_Score', 'Operation_Time', 'GFR']

# 定义分类特征（二分类）
original_categorical = ['Weakened', 'Depression', 'Nutritional_Risk']

# 用户输入
input_data_original = {}

# 连续特征输入
for col in continuous_cols:
    if col == 'Education':
        input_data_original[col] = st.selectbox(f"{col} (0=文盲, 1=小学, 2=初中, 3=高中, 4=大专及以上):", 
                                                 options=list(range(0, 5)))
    else:
        input_data_original[col] = st.number_input(f"{col}:", value=0.0)

# 分类特征输入
for col in original_categorical:
    input_data_original[col] = st.selectbox(f"{col}:", options=[0, 1])

# =========================
# 3. 预测按钮
# =========================
if st.button("predict"):
    # 构建输入数据
    input_data = {}
    
    # 设置连续特征
    for col in continuous_cols:
        if col in feature_cols:
            input_data[col] = input_data_original[col]
    
    # 设置分类特征
    for col in original_categorical:
        if col in feature_cols:
            input_data[col] = input_data_original[col]
    
    # 创建DataFrame
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # 标准化连续特征
    X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])
    
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # =========================
    # 创建背景数据 - 所有特征设为参考值
    # =========================
    background_data = {}
    
    # 连续特征设为 0（或中位数）
    for col in continuous_cols:
        background_data[col] = 0
    
    # 分类特征设为 0
    for col in original_categorical:
        if col in feature_cols:
            background_data[col] = 0
    
    background_data = pd.DataFrame([background_data], columns=feature_cols)
    
    # 对背景数据的连续特征也进行标准化
    background_data[continuous_cols] = scaler.transform(background_data[continuous_cols])

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # 显示所有特征的 SHAP 值
    # =========================
    all_features = continuous_cols + original_categorical
    
    st.write("SHAP values of each feature：")
    for i, name in enumerate(feature_cols):
        if name in all_features:
            st.write(f"{name}: {shap_values[i]:.4f} (value = {input_data_original[name]})")

    # =========================
    # 生成 HTML force plot
    # =========================
    st.subheader("SHAP force plot of the prediction")
    
    # 准备显示用的特征名和值
    display_features = []
    display_values = []
    display_shap = []
    
    for name in feature_cols:
        if name in all_features:
            display_features.append(name)
            display_values.append(input_data_original[name])
            display_shap.append(shap_values[feature_cols.index(name)])
    
    # 生成 force plot
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=np.array(display_shap),
        features=np.array(display_values),
        feature_names=display_features,
        matplotlib=False
    )
    
    shap.save_html("shap_force_plot.html", force_plot)
    
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=300, scrolling=False)
