import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    # 自定义 Force Plot 可视化（显示所有特征）
    # =========================
    st.subheader("SHAP force plot of the prediction")
    
    base_value = explainer.expected_value
    shap_vals = [aggregated_shap[f] for f in original_features]
    feature_vals = [aggregated_values[f] for f in original_features]
    
    # 按 SHAP 值排序
    sorted_indices = np.argsort(shap_vals)
    sorted_features = [original_features[i] for i in sorted_indices]
    sorted_shap = [shap_vals[i] for i in sorted_indices]
    sorted_values = [feature_vals[i] for i in sorted_indices]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # 计算累积位置
    cumsum = np.cumsum([0] + sorted_shap)
    
    # 绘制每个特征的贡献
    for i, (feat, shap_val, feat_val) in enumerate(zip(sorted_features, sorted_shap, sorted_values)):
        color = '#ff0051' if shap_val > 0 else '#008bfb'
        
        # 绘制矩形
        rect = mpatches.FancyBboxPatch(
            (cumsum[i] + base_value, 0.3),
            shap_val,
            0.4,
            boxstyle="round,pad=0.01",
            linewidth=1,
            edgecolor='white',
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # 添加特征标签
        text_x = cumsum[i] + base_value + shap_val / 2
        ax.text(text_x, 0.5, f'{feat}={feat_val}', 
                ha='center', va='center', fontsize=8, 
                color='white', weight='bold', rotation=0)
    
    # 设置坐标轴
    final_value = base_value + sum(sorted_shap)
    ax.set_xlim(min(base_value, final_value) - 0.05, max(base_value, final_value) + 0.05)
    ax.set_ylim(0, 1)
    
    # 添加基准值和输出值标签
    ax.text(base_value, 0.1, f'base value\n{base_value:.3f}', 
            ha='center', va='top', fontsize=10, weight='bold')
    ax.text(final_value, 0.1, f'f(x)\n{final_value:.3f}', 
            ha='center', va='top', fontsize=10, weight='bold')
    
    # 添加箭头
    ax.annotate('', xy=(final_value, 0.5), xytext=(base_value, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 隐藏坐标轴
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Model output value', fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
