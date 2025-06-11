# pages/1_📊_Gambaran_Dataset.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Gambaran Dataset", layout="wide")
st.title("📊 Gambaran Umum Dataset Telco Customer Churn")
st.write("Halaman ini menampilkan analisis dari dataset customer churn perusahaan telekomunikasi melalui Exploratory Data Analysis (EDA).")

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di {file_path}. Pastikan file 'Churn.csv' ada di folder 'data'.")
        return None

# Muat data
data_path = os.path.join('data', 'Churn.csv') 
df = load_data(data_path)

if df is not None:
    # --- Informasi Dataset ---
    st.header("📋 Informasi Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pelanggan", f"{df.shape[0]:,}")
        
    with col2:
        st.metric("Total Fitur", df.shape[1])
        
    with col3:
        churn_count = df['Churn'].value_counts()['Yes']
        churn_pct = (churn_count / len(df)) * 100
        st.metric("Pelanggan Churn", f"{churn_count:,} ({churn_pct:.1f}%)")
        
    with col4:
        non_churn_count = df['Churn'].value_counts()['No']
        non_churn_pct = (non_churn_count / len(df)) * 100
        st.metric("Pelanggan Setia", f"{non_churn_count:,} ({non_churn_pct:.1f}%)")

    # --- Pratinjau Data ---
    st.header("🔍 Pratinjau Data")
    st.subheader("Berikut 5 baris data teratas:")
    st.dataframe(df.head(), use_container_width=True)

    # --- Deskripsi Fitur ---
    st.header("📝 Deskripsi Fitur")
    
    st.markdown("""
    Dataset berisi informasi pelanggan telekomunikasi dengan fitur-fitur:
    
    **Demografis:** Gender, SeniorCitizen, Partner, Dependents  
    **Layanan:** Tenure, Internet Service, Phone Service, Streaming Services  
    **Account:** Contract, Payment Method, Monthly/Total Charges  
    **Target:** Churn (Yes/No) - Variable yang diprediksi
    """)

    st.markdown("---")

    # --- Visualisasi ---
    st.header("📊 Visualisasi & Analisis")    # 1. Distribusi Churn
    st.subheader("1. Distribusi Target Variable (Churn)")
    
    fig_churn, ax_churn = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, ax=ax_churn, palette=['#2E86AB', '#A23B72'])
    ax_churn.set_title('Distribusi Pelanggan Churn vs Non-Churn', fontsize=12, fontweight='bold')
    ax_churn.set_ylabel('Jumlah Pelanggan')
    ax_churn.set_xlabel('Status Churn')
    
    # Tambahkan label persentase
    total = len(df)
    for i, p in enumerate(ax_churn.patches):
        height = p.get_height()
        percentage = 100 * height / total
        ax_churn.text(p.get_x() + p.get_width()/2., height + 20, 
                     f'{int(height)}\n({percentage:.1f}%)', 
                     ha="center", va="bottom", fontweight='bold', fontsize=10)
    
    # Set y-axis limit untuk memberikan ruang bagi label
    ax_churn.set_ylim(0, max([p.get_height() for p in ax_churn.patches]) * 1.15)
    
    # Tambahkan grid untuk kemudahan membaca
    ax_churn.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_churn)
    
    st.warning("⚠️ **Dataset Tidak Seimbang**: 73.5% pelanggan tidak churn vs 26.5% churn. Akan ditangani dengan teknik SMOTE.")

    # 2. Distribusi Fitur Numerik
    st.subheader("2. Distribusi Fitur Numerik Utama")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_tenure, ax_tenure = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=20, ax=ax_tenure, palette=['#2E86AB', '#A23B72'])
        ax_tenure.set_title('Distribusi Tenure berdasarkan Status Churn')
        ax_tenure.set_xlabel('Tenure (bulan)')
        st.pyplot(fig_tenure)
    
    with col2:
        fig_monthly, ax_monthly = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, bins=20, ax=ax_monthly, palette=['#2E86AB', '#A23B72'])
        ax_monthly.set_title('Distribusi Monthly Charges berdasarkan Status Churn')
        ax_monthly.set_xlabel('Monthly Charges ($)')
        st.pyplot(fig_monthly)
    
    st.info("""
    **Key Insights:**
    - Pelanggan dengan **tenure rendah** lebih berisiko churn
    - Pelanggan dengan **biaya bulanan tinggi** cenderung churn
    - Pelanggan setia umumnya memiliki tenure panjang dan biaya stabil
    """)

    # 3. Analisis Fitur Kategorikal
    st.subheader("3. Analisis Fitur Kategorikal Kunci")
    
    categorical_features = ['Contract', 'InternetService', 'PaymentMethod']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Churn Rate berdasarkan Fitur Kategorikal', fontsize=14, fontweight='bold')
    
    for i, feature in enumerate(categorical_features):
        ax = axes[i]
        ct = pd.crosstab(df[feature], df['Churn'], normalize='index') * 100
        ct.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'])
        ax.set_title(f'Churn Rate by {feature}')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['No Churn', 'Churn'])
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.success("""
    **Key Findings:**
    - **Contract**: Month-to-month memiliki churn rate tertinggi
    - **Internet Service**: Fiber optic users lebih berisiko churn  
    - **Payment Method**: Electronic check berkaitan dengan churn tinggi
    """)

    st.markdown("---")
    
    # Kesimpulan
    st.header("🎯 Kesimpulan EDA")
    
    st.markdown("""
    **📊 Karakteristik Dataset:**
    - 7,043 pelanggan dengan 21 fitur
    - Dataset tidak seimbang (73.5% vs 26.5%)
    
    **🔍 Faktor Utama Churn:**
    1. **Tenure rendah** - Pelanggan baru berisiko tinggi
    2. **Contract month-to-month** - Fleksibilitas tinggi = churn tinggi  
    3. **Biaya bulanan tinggi** - Berdampak pada kepuasan
    4. **Payment method** - Electronic check bermasalah
    """)

else:
    st.error("❌ Gagal memuat dataset. Pastikan file 'Churn.csv' tersedia di folder 'data'.")