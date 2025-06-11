# pages/2_🏋️_Pelatihan_Model.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Pelatihan Model", 
    layout="wide",
    page_icon="🏋️",
    initial_sidebar_state="expanded"
)

# --- Header dan Deskripsi ---
st.title("🏋️ Pelatihan dan Evaluasi Model")
st.markdown("Perbandingan performa 4 model Machine Learning untuk prediksi churn pelanggan")
st.markdown("---")

# --- Metodologi Training ---
st.markdown("### 🔬 Metodologi Pelatihan")

with st.expander("📖 Detail Proses Training", expanded=False):
    st.markdown("""
    **1. Preprocessing Data:**
    - Data cleaning dan handling missing values
    - Feature encoding untuk variabel kategorikal
    - Standard scaling untuk normalisasi fitur numerik
    
    **2. Handling Imbalanced Data:**
    - Menggunakan **SMOTE** (Synthetic Minority Oversampling Technique)
    - Menyeimbangkan distribusi kelas target (Churn vs No Churn)
    
    **3. Model Training:**
    - Split data: 80% training, 20% testing
    - Cross-validation untuk validasi model
    - Hyperparameter tuning untuk optimisasi performa
    
    **4. Evaluasi Model:**
    - Metrik utama: Accuracy, Precision, Recall, F1-Score
    - Focus pada deteksi kelas minority (Churn = Yes)
    """)

# --- Informasi Training ---
st.markdown("### 📋 Overview Training")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #e3f2fd; text-align: center;">
        <h3 style="margin: 0; color: #1976d2;">📊 Dataset</h3>
        <p style="margin: 5px 0; color: #1976d2; font-size: 18px; font-weight: bold;">Telco Churn</p>
        <p style="margin: 0; color: #666; font-size: 12px;">7043 samples</p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #f3e5f5; text-align: center;">
        <h3 style="margin: 0; color: #7b1fa2;">⚖️ Balancing</h3>
        <p style="margin: 5px 0; color: #7b1fa2; font-size: 18px; font-weight: bold;">SMOTE</p>
        <p style="margin: 0; color: #666; font-size: 12px;">Synthetic Oversampling</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #fff3e0; text-align: center;">
        <h3 style="margin: 0; color: #f57c00;">🤖 Models</h3>
        <p style="margin: 5px 0; color: #f57c00; font-size: 18px; font-weight: bold;">4 Algoritma</p>
        <p style="margin: 0; color: #666; font-size: 12px;">ML Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
with col4:
    st.markdown("""
    <div style="padding: 15px; border-radius: 10px; background-color: #e8f5e8; text-align: center;">
        <h3 style="margin: 0; color: #388e3c;">🏆 Winner</h3>
        <p style="margin: 5px 0; color: #388e3c; font-size: 18px; font-weight: bold;">XGBoost</p>
        <p style="margin: 0; color: #666; font-size: 12px;">Best F1-Score</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Ringkasan Metrik Kinerja ---
st.markdown("### 📈 Perbandingan Performa Model")

# Data diambil langsung dari hasil Anda
data_performa = {
    'Model': ['XGBoost', 'Logistic Regression', 'Random Forest', 'Support Vector Machine'],
    'Akurasi': [0.8009, 0.7495, 0.7802, 0.7545],
    'Precision (untuk Churn=Yes)': [0.61, 0.52, 0.58, 0.53],
    'Recall (untuk Churn=Yes)': [0.68, 0.80, 0.59, 0.72],
    'F1-score (untuk Churn=Yes)': [0.64, 0.63, 0.59, 0.61]
}
df_performa = pd.DataFrame(data_performa).sort_values(by='F1-score (untuk Churn=Yes)', ascending=False).reset_index(drop=True)

# Membuat ranking berdasarkan F1-score
df_performa['Ranking'] = ['🥇', '🥈', '🥉', '4️⃣']
df_performa = df_performa[['Ranking', 'Model', 'Akurasi', 'Precision (untuk Churn=Yes)', 'Recall (untuk Churn=Yes)', 'F1-score (untuk Churn=Yes)']]

# Styling untuk highlight nilai tertinggi di setiap kolom
def highlight_max(s):
    """
    Highlight nilai maksimum dalam setiap kolom numerik.
    """
    if s.name in ['Ranking', 'Model']:  # Skip kolom non-numerik
        return [''] * len(s)
    else:
        is_max = s == s.max()
        return ['background-color: #4caf50; color: white; font-weight: bold' if v else '' for v in is_max]

styled_df = df_performa.style.apply(highlight_max, axis=0).format({
    'Akurasi': '{:.3f}',
    'Precision (untuk Churn=Yes)': '{:.2f}',
    'Recall (untuk Churn=Yes)': '{:.2f}',
    'F1-score (untuk Churn=Yes)': '{:.2f}'
})

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# --- Visualisasi Perbandingan Metrik ---
st.markdown("### 📊 Visualisasi Performa")

# Create subplot dengan 2 kolom
col1, col2 = st.columns(2)

with col1:
    # Bar chart untuk Akurasi
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 5))
    bars1 = ax_accuracy.bar(df_performa['Model'], df_performa['Akurasi'], 
                           color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax_accuracy.set_title('Perbandingan Akurasi Model', fontweight='bold', fontsize=12)
    ax_accuracy.set_ylabel('Akurasi')
    ax_accuracy.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Tambahkan label nilai
    for bar in bars1:
        height = bar.get_height()
        ax_accuracy.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_accuracy)

with col2:
    # Bar chart untuk F1-score
    fig_f1, ax_f1 = plt.subplots(figsize=(8, 5))
    bars2 = ax_f1.bar(df_performa['Model'], df_performa['F1-score (untuk Churn=Yes)'], 
                      color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax_f1.set_title('Perbandingan F1-Score Model', fontweight='bold', fontsize=12)
    ax_f1.set_ylabel('F1-Score')
    ax_f1.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')    # Tambahkan label nilai
    for bar in bars2:
        height = bar.get_height()
        ax_f1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_f1)

st.info("""
**Interpretasi Tabel:**
- **XGBoost** menunjukkan **F1-score** (0.64) dan **Akurasi** (0.8009) tertinggi, menjadikannya model yang paling seimbang dan akurat secara keseluruhan untuk prediksi churn.
- **Logistic Regression** memiliki **Recall** tertinggi (0.80), yang berarti model ini paling baik dalam mendeteksi semua pelanggan yang berpotensi churn, namun dengan precision yang lebih rendah.
- **Random Forest** menunjukkan performa yang stabil dengan akurasi kedua tertinggi (0.7802), tetapi recall yang rendah (0.59) untuk mendeteksi pelanggan churn.
- **Support Vector Machine** memiliki performa yang seimbang dengan recall tinggi (0.72) untuk churn, tetapi akurasi keseluruhan yang lebih rendah (0.7545).

**Rekomendasi:** XGBoost dipilih sebagai model terbaik karena memberikan keseimbangan optimal antara precision dan recall.
""")
st.markdown("---")

# --- Laporan Klasifikasi Detail ---
st.header("Laporan Klasifikasi Detail untuk Setiap Model")
tab1, tab2, tab3, tab4 = st.tabs(["XGBoost", "Logistic Regression", "Random Forest", "Support Vector Machine"])

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