import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt

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

# 定义原始分类特征
original_categorical = ['Education', 'Weakened', 'Depression', 'Nutritional_Risk']

# 用户输入原始值
input_data_original = {}
for col in continuous_cols:
    input_data_original[col] = st.number_input(f"{col}:", value=0.0)

for col in original_categorical:
    if col == 'Education':
        input_data_original[col] = st.selectbox(f"{col}:", options=list(range(0, 5)))
    else:
        input_data_original[col] = st.selectbox(f"{col}:", options=[0, 1])

# =========================
# 3. 预测按钮
# =========================
if st.button("predict"):
    # 初始化所有特征为0
    input_data = {col: 0 for col in feature_cols}
    
    # 设置连续特征
    for col in continuous_cols:
        if col in feature_cols:
            input_data[col] = input_data_original[col]
    
    # 设置分类特征（独热编码）
    for orig_col in original_categorical:
        user_value = input_data_original[orig_col]
        related_cols = [col for col in feature_cols if col.startswith(orig_col + '_')]
        for col in related_cols:
            try:
                category = int(col.split('_')[-1])
                input_data[col] = 1 if user_value == category else 0
            except:
                pass
    
    # 创建DataFrame，确保列顺序与feature_cols一致
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # 标准化连续特征
    X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])
    
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # 创建背景数据
    background_data = pd.DataFrame([{col: 0 for col in feature_cols}], columns=feature_cols)

    explainer = shap.KernelExplainer(
        model=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_cols))[:, 1],
        data=background_data,
        link="identity"
    )

    shap_values = explainer.shap_values(X_input, nsamples=100)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    # =========================
    # 聚合 SHAP 值到原始特征
    # =========================
    original_features = continuous_cols + original_categorical
    aggregated_shap = {}
    aggregated_values = {}

    for orig_feat in original_features:
        if orig_feat in continuous_cols:
            idx = feature_cols.index(orig_feat)
            aggregated_shap[orig_feat] = shap_values[idx]
            aggregated_values[orig_feat] = input_data_original[orig_feat]
        else:
            related_cols = [col for col in feature_cols if col.startswith(orig_feat + '_')]
            related_shap = sum([shap_values[feature_cols.index(col)] for col in related_cols])
            aggregated_shap[orig_feat] = related_shap
            aggregated_values[orig_feat] = input_data_original[orig_feat]

    # 显示原始特征的 SHAP 值
    st.write("SHAP values of each feature (original features)：")
    for name in original_features:
        st.write(f"{name}: {aggregated_shap[name]:.4f}")

    # =========================
    # 自定义条形图显示所有特征
    # =========================
    st.subheader("SHAP values visualization (all features)")
    
    shap_vals = [aggregated_shap[f] for f in original_features]
    feature_vals = [aggregated_values[f] for f in original_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建颜色：正值红色，负值蓝色
    colors = ['#ff0051' if val > 0 else '#008bfb' for val in shap_vals]
    
    # 绘制水平条形图
    y_pos = np.arange(len(original_features))
    ax.barh(y_pos, shap_vals, color=colors, alpha=0.8)
    
    # 设置标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name} = {feature_vals[i]}" for i, name in enumerate(original_features)])
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
    ax.set_title('Feature Contributions to Prediction', fontsize=14, fontweight='bold')
    
    # 添加基准线
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
