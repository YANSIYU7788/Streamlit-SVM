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

# 显示实际的特征列，帮助调试
st.sidebar.write("Model features:")
st.sidebar.write(feature_cols)

# =========================
# 2. 自动检测 Education 是连续变量还是独热编码
# =========================
education_is_continuous = 'Education' in feature_cols
education_cols = [col for col in feature_cols if col.startswith('Education_')]

if education_is_continuous:
    st.sidebar.write("Education: 连续变量")
    continuous_cols = ['Age', 'Education', 'MOCA_Score', 'Operation_Time', 'GFR']
    continuous_cols = [col for col in continuous_cols if col in feature_cols]
else:
    st.sidebar.write("Education: 独热编码")
    continuous_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']
    continuous_cols = [col for col in continuous_cols if col in feature_cols]

# 识别分类特征
categorical_features = []
for prefix in ['Weakened', 'Depression', 'Nutritional_Risk']:
    if prefix in feature_cols:
        categorical_features.append(prefix)
    elif any(col.startswith(prefix + '_') for col in feature_cols):
        categorical_features.append(prefix)

# =========================
# 3. Streamlit 界面
# =========================
st.title("SVM predict dNCR")
st.write("Please enter patient characteristics：")

input_data_original = {}

# 连续特征输入
for col in ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']:
    if col in feature_cols:
        input_data_original[col] = st.number_input(f"{col}:", value=0.0)

# Education 输入
if education_is_continuous:
    input_data_original['Education'] = st.selectbox(
        "Education (0=文盲, 1=小学, 2=初中, 3=高中, 4=大专及以上):", 
        options=list(range(0, 5))
    )
else:
    input_data_original['Education'] = st.selectbox(
        "Education (0=文盲, 1=小学, 2=初中, 3=高中, 4=大专及以上):", 
        options=list(range(0, 5))
    )

# 其他分类特征输入
for feat in ['Weakened', 'Depression', 'Nutritional_Risk']:
    if feat in categorical_features:
        input_data_original[feat] = st.selectbox(f"{feat}:", options=[0, 1])

# =========================
# 4. 预测按钮
# =========================
if st.button("predict"):
    # 构建输入数据
    input_data = {}
    
    # 处理连续特征
    for col in continuous_cols:
        input_data[col] = input_data_original[col]
    
    # 处理 Education（如果是独热编码）
    if not education_is_continuous:
        for col in education_cols:
            category = int(col.split('_')[-1])
            input_data[col] = 1 if input_data_original['Education'] == category else 0
    
    # 处理其他分类特征
    for feat in categorical_features:
        if feat in feature_cols:
            # 直接作为 0/1
            input_data[feat] = input_data_original[feat]
        else:
            # 独热编码
            related_cols = [col for col in feature_cols if col.startswith(feat + '_')]
            for col in related_cols:
                category = int(col.split('_')[-1])
                input_data[col] = 1 if input_data_original[feat] == category else 0
    
    # 创建DataFrame
    X_input = pd.DataFrame([input_data], columns=feature_cols)
    
    # 标准化连续特征
    if len(continuous_cols) > 0:
        X_input[continuous_cols] = scaler.transform(X_input[continuous_cols])
    
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"Predicted probabilities: {pred_prob[0]:.4f}")
    st.write(f"Predicted results: {'yes' if pred_label[0] == 1 else 'no'}")

    # =========================
    # 创建背景数据
    # =========================
    background_data = {col: 0 for col in feature_cols}
    
    # 设置分类特征的参考类别
    if not education_is_continuous and education_cols:
        background_data['Education_0'] = 1
    
    for feat in categorical_features:
        if feat not in feature_cols:
            related_cols = [col for col in feature_cols if col.startswith(feat + '_')]
            if related_cols:
                background_data[f'{feat}_0'] = 1
    
    background_data = pd.DataFrame([background_data], columns=feature_cols)
    
    # 标准化背景数据的连续特征
    if len(continuous_cols) > 0:
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
    # 聚合 SHAP 值
    # =========================
    display_features = ['Age', 'Education', 'MOCA_Score', 'Operation_Time', 'GFR', 
                       'Weakened', 'Depression', 'Nutritional_Risk']
    aggregated_shap = {}
    aggregated_values = {}
    
    for feat in display_features:
        if feat in feature_cols:
            # 直接存在的特征
            idx = feature_cols.index(feat)
            aggregated_shap[feat] = shap_values[idx]
            aggregated_values[feat] = input_data_original.get(feat, 0)
        else:
            # 独热编码的特征，需要聚合
            related_cols = [col for col in feature_cols if col.startswith(feat + '_')]
            if related_cols:
                related_shap = sum([shap_values[feature_cols.index(col)] for col in related_cols])
                aggregated_shap[feat] = related_shap
                aggregated_values[feat] = input_data_original.get(feat, 0)

    # 显示 SHAP 值
    st.write("SHAP values of each feature：")
    for name in display_features:
        if name in aggregated_shap:
            st.write(f"{name}: {aggregated_shap[name]:.4f} (value = {aggregated_values[name]})")

    # =========================
    # 生成 force plot
    # =========================
    st.subheader("SHAP force plot of the prediction")
    
    valid_features = [f for f in display_features if f in aggregated_shap]
    shap_vals_array = np.array([aggregated_shap[f] for f in valid_features])
    feature_vals_array = np.array([aggregated_values[f] for f in valid_features])
    
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_vals_array,
        features=feature_vals_array,
        feature_names=valid_features,
        matplotlib=False
    )
    
    shap.save_html("shap_force_plot.html", force_plot)
    
    with open("shap_force_plot.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    st.components.v1.html(html_content, height=300, scrolling=False)
