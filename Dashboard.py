# app.py
import streamlit as st

st.set_page_config(
    page_title="Dashboard Prediksi Churn",
    page_icon="👋",
    layout="wide"
)

st.title("Selamat Datang di Dashboard Prediksi Churn Pelanggan!")
st.write("""
Proyek ini bertujuan untuk menganalisis faktor-faktor yang mempengaruhi churn pelanggan pada sebuah perusahaan telekomunikasi.
Berdasarkan analisis tersebut, kami membangun beberapa model machine learning untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (churn).
""")
st.write("Silakan pilih halaman dari sidebar di sebelah kiri untuk memulai:")
st.markdown("""
- **📊 Gambaran Dataset**: Melihat karakteristik dan visualisasi data mentah.
- **🏋️ Pelatihan dan Evaluasi Model**: Melihat performa model machine learning yang telah dilatih.
- **🔮 Prediksi Churn**: Mencoba memprediksi churn pelanggan berdasarkan input fitur.
""")

st.sidebar.success("Pilih halaman di atas.")