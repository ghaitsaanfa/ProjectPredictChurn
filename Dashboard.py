import streamlit as st

st.set_page_config(
    page_title="Dashboard Prediksi Churn",
    page_icon="👋",
    layout="wide"
)

# Logo (opsional)
# st.image("link_logo.png", width=100)

st.title("Selamat Datang di Dashboard Prediksi Churn Pelanggan! 👋")

st.info("""
Proyek ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi **churn pelanggan** pada sebuah perusahaan telekomunikasi.
Berdasarkan analisis tersebut, beberapa model machine learning dibangun untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (**churn**).
""")

st.markdown("---")

kol1, kol2 = st.columns([2, 3])
with kol1:
    st.subheader("🔎 Fitur Dashboard")
    st.markdown("""
    - **📊 Gambaran Dataset**: Melihat karakteristik dan visualisasi data mentah.
    - **🏋️ Pelatihan & Evaluasi Model**: Analisis performa model machine learning.
    - **🔮 Prediksi Churn**: Coba prediksi churn pelanggan berdasarkan input fitur.
    """)
    st.success("Pilih halaman dari sidebar untuk mulai eksplorasi!")

with kol2:
    st.subheader("Tentang Churn Pelanggan")
    st.write("""
    Churn pelanggan adalah kondisi di mana pelanggan berhenti menggunakan layanan/perusahaan. 
    Dashboard ini membantu perusahaan untuk:
    - Mengidentifikasi faktor utama penyebab churn.
    - Melakukan prediksi terhadap pelanggan yang berpotensi churn.
    - Membantu pengambilan keputusan bisnis berbasis data.
    """)

st.markdown("---")
st.sidebar.header("Navigasi")
st.sidebar.success("Pilih halaman di atas.")

# Nama Kelompok
st.markdown("<center>### Kelompok 2 - Data Mining")
# Footer dengan nama kelompok
st.markdown(
    "<hr><center><span style='color:gray'>© 2025 Kelompok 2 - Data Mining | Universitas Negeri Semarang</span></center>",
    unsafe_allow_html=True
)
