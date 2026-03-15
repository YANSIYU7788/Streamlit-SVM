import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# 1. 加载模型、标准化器和特征列
# =========================
model = joblib.load("D:/JQXX/svm_final_model.pkl")
scaler = joblib.load("D:/JQXX/svm_scaler.pkl")
feature_cols = joblib.load("D:/JQXX/svm_feature_columns.pkl")

# =========================
# 2. Streamlit 界面
# =========================
st.title("SVM 预测 dNCR")

st.write("请输入患者特征：")

# 例子：生成输入框
input_data = {}
for col in feature_cols:
    if col in ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']:
        input_data[col] = st.number_input(f"{col}:", value=0)
    else:
        input_data[col] = st.selectbox(f"{col}:", options=[0,1])

# 转成 DataFrame
X_input = pd.DataFrame([input_data])

# 标准化连续变量
scale_cols = ['Age', 'MOCA_Score', 'Operation_Time', 'GFR']
X_input[scale_cols] = scaler.transform(X_input[scale_cols])

# =========================
# 3. 预测按钮
# =========================
if st.button("预测"):
    pred_prob = model.predict_proba(X_input)[:, 1]
    threshold = 0.16   # 你的最佳阈值
    pred_label = (pred_prob >= threshold).astype(int)

    st.write(f"预测概率: {pred_prob[0]:.4f}")
    st.write(f"预测结果: {'yes' if pred_label[0]==1 else 'no'}")
