import streamlit as st

st.set_page_config(
    page_title="Dashboard Prediksi Churn",
    page_icon="🏫",
    layout="wide"
)

# --- Custom CSS agar selaras ---
st.markdown("""
    <style>
        .main { background-color: #f6f9fb; }
        h1 { color: #2E86AB !important; }
        h2, h3 { color: #e3f2fd !important; }
        .stMetric-value { color: #2E86AB !important; }
        .stMetric-label { color: #555 !important; }
        .stTabs [data-baseweb="tab"] { font-size:17px; padding: 12px 20px; }
        .stDataFrame th { background-color: #e3f2fd !important; }
    </style>
""", unsafe_allow_html=True)

# Banner/logo atas
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <h1 style="margin: 0; color: #2E86AB;">Selamat Datang di Dashboard Prediksi Churn Pelanggan!👋</h1>
    </div>
    """, unsafe_allow_html=True
)

st.info("""
Proyek ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi **churn pelanggan** pada perusahaan telekomunikasi.
Berdasarkan analisis tersebut, beberapa model machine learning dibangun untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (**churn**).
""", icon="ℹ️")

st.markdown("---")

kol1, kol2 = st.columns([2, 3])
with kol1:
    st.subheader("🔎 Fitur Dashboard")
    st.markdown("""
    - **📊 Gambaran Dataset**  
      Melihat karakteristik dan visualisasi data mentah.
    - **🏋️ Pelatihan & Evaluasi Model**  
      Analisis performa model machine learning.
    - **🔮 Prediksi Churn**  
      Coba prediksi churn pelanggan berdasarkan input fitur.
    """)
    st.success("Pilih halaman dari sidebar untuk mulai eksplorasi!", icon="✅")

with kol2:
    st.subheader("📖 Tentang Churn Pelanggan")
    st.markdown("""
    Churn pelanggan adalah kondisi di mana pelanggan berhenti menggunakan layanan/perusahaan.  
    Dashboard ini membantu perusahaan untuk:
    - Mengidentifikasi faktor utama penyebab churn.
    - Melakukan prediksi terhadap pelanggan yang berpotensi churn.
    - Membantu pengambilan keputusan bisnis berbasis data.
    """)

st.markdown("---")
st.sidebar.header("Navigasi")
st.sidebar.success("Pilih halaman di atas.")

st.markdown("### Kelompok 2 - Data Mining")
st.markdown(
    "<hr><center><span style='color: #999;'>© 2025 Kelompok 2 - Data Mining | Universitas Negeri Semarang</span></center>",
    unsafe_allow_html=True
)
