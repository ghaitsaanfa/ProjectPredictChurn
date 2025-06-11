import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Pelatihan & Evaluasi Model | Churn Prediction", 
    layout="wide",
    page_icon="🏋️"
)

# --- Custom CSS for better aesthetics ---
st.markdown("""
    <style>
        .main { background-color: #f6f9fb; }
        h1, h2, h3 { color: #304ffe; }
        .stDataFrame th { background-color: #e3f2fd !important; }
        .stTabs [data-baseweb="tab"] { font-size:17px; padding: 12px 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🏋️ Pelatihan & Evaluasi Model")
st.caption("Project Akhir Data Mining Kelompok 2 | Prediksi Churn Pelanggan Telco")
st.markdown("---")

# --- Training Methodology Section ---
with st.container():
    st.subheader("🔬 Metodologi Pelatihan")
    st.info("""
    **1. Preprocessing:** Data cleaning, feature encoding, dan normalisasi.
    **2. Handling Imbalanced Data:** Menggunakan SMOTE.
    **3. Model Training:** 80/20 split, cross-validation, hyperparameter tuning.
    **4. Evaluasi:** Akurasi, Precision, Recall, F1-Score (fokus pada deteksi churn).
    """, icon="⚙️")

# --- Overview Cards ---
st.markdown("### 📋 Overview Training")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Dataset", value="Telco Churn", delta="7043 samples", help="Jumlah data setelah preprocessing")
with col2:
    st.metric(label="Balancing", value="SMOTE", delta="Synthetic Oversampling")
with col3:
    st.metric(label="Model", value="4 Algoritma", delta="ML Classification")
with col4:
    st.metric(label="🏆 Best Model", value="XGBoost", delta="Best F1-Score")

st.divider()

# --- Performance Table ---
st.markdown("### 📈 Perbandingan Performa Model")
data_performa = {
    'Model': ['XGBoost', 'Logistic Regression', 'Random Forest', 'Support Vector Machine'],
    'Akurasi': [0.8009, 0.7495, 0.7802, 0.7545],
    'Precision (Churn)': [0.61, 0.52, 0.58, 0.53],
    'Recall (Churn)': [0.68, 0.80, 0.59, 0.72],
    'F1-Score (Churn)': [0.64, 0.63, 0.59, 0.61]
}
df_performa = pd.DataFrame(data_performa).sort_values('F1-Score (Churn)', ascending=False).reset_index(drop=True)
df_performa['Ranking'] = ['🥇', '🥈', '🥉', '4️⃣']
df_performa = df_performa[['Ranking','Model','Akurasi','Precision (Churn)','Recall (Churn)','F1-Score (Churn)']]

def highlight_max(s):
    if s.name in ['Ranking', 'Model']: return ['']*len(s)
    is_max = s == s.max()
    return ['background-color: #304ffe; color: white; font-weight: bold' if v else '' for v in is_max]

st.dataframe(
    df_performa.style.apply(highlight_max, axis=0).format({
        'Akurasi': '{:.3f}',
        'Precision (Churn)': '{:.2f}',
        'Recall (Churn)': '{:.2f}',
        'F1-Score (Churn)': '{:.2f}'
    }),
    use_container_width=True, hide_index=True
)

# --- Performance Visualization ---
st.markdown("### 📊 Visualisasi Performa Model")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Akurasi', data=df_performa, palette='Blues_r', ax=ax)
    ax.set_title('Akurasi Tiap Model')
    ax.set_ylim(0,1)
    st.pyplot(fig, use_container_width=True)
with col2:
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='F1-Score (Churn)', data=df_performa, palette='Purples', ax=ax)
    ax.set_title('F1-Score (Churn) Tiap Model')
    ax.set_ylim(0,1)
    st.pyplot(fig, use_container_width=True)

# --- Interpretation & Recommendation ---
with st.expander("📖 Interpretasi Hasil & Rekomendasi", expanded=True):
    st.success("""
    - **XGBoost**: F1-score tertinggi (0.64), akurasi terbaik (0.80) → **Recommended**.
    - **Logistic Regression**: Recall tertinggi (0.80), baik untuk mendeteksi churn.
    - **Random Forest**: Akurasi tinggi, recall rendah.
    - **SVM**: Recall tinggi (0.72), akurasi lebih rendah.
    """)

# --- Classification Report Tabs ---
st.divider()
st.markdown("### 📄 Laporan Klasifikasi Detail")
tab1, tab2, tab3, tab4 = st.tabs(["XGBoost", "Logistic Regression", "Random Forest", "SVM"])
with tab1:
    st.subheader("XGBoost")
    st.code("""
              precision    recall  f1-score   support

          No       0.88      0.84      0.86      1031
         Yes       0.61      0.68      0.64       370

    accuracy                           0.80      1401
    """, language='text')
with tab2:
    st.subheader("Logistic Regression")
    st.code("""
              precision    recall  f1-score   support

          No       0.91      0.73      0.81      1031
         Yes       0.52      0.80      0.63       370

    accuracy                           0.75      1401
    """, language='text')
with tab3:
    st.subheader("Random Forest")
    st.code("""
              precision    recall  f1-score   support

          No       0.85      0.85      0.85      1031
         Yes       0.58      0.59      0.59       370

    accuracy                           0.78      1401
    """, language='text')
with tab4:
    st.subheader("Support Vector Machine")
    st.code("""
              precision    recall  f1-score   support

          No       0.88      0.77      0.82      1031
         Yes       0.53      0.72      0.61       370

    accuracy                           0.75      1401
    """, language='text')

# --- Footer / Divider ---
st.divider()
st.markdown("<center><span style='color: #999;'>© 2025 Kelompok 2 Data Mining</span></center>", unsafe_allow_html=True)
