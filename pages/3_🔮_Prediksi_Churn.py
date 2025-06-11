import streamlit as st
import pandas as pd
import joblib
import os

# --- HEADER & LOGO (Selaraskan dengan Page 1 & 2) ---
st.set_page_config(
    page_title="Prediksi Churn", 
    layout="wide",
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Selaraskan dengan Page 1 & 2) ---
st.markdown("""
    <style>
        .main { background-color: #f6f9fb; }
        h1 { color: #2E86AB !important; }
        h2, h3, h4 { color: #e3f2fd !important; }
        .stDataFrame th { background-color: #e3f2fd !important; }
        .stTabs [data-baseweb="tab"] { font-size:17px; padding: 12px 20px; }
        .stMetric-value { color: #2E86AB !important; }
        .stMetric-label { color: #555 !important; }
        .stDownloadButton { background-color: #2E86AB !important; color: white !important; }
        .stButton>button { border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <img src="https://cdn-icons-png.flaticon.com/512/2203/2203187.png" width="50" style="margin-right: 20px;">
        <h1 style="margin: 0; color: #2E86AB;">Prediksi Churn Pelanggan</h1>
    </div>
    """, unsafe_allow_html=True
)

# --- INFO BOX ---
st.info(
    "Halaman ini menggunakan model Machine Learning yang telah dilatih untuk memprediksi kemungkinan pelanggan berhenti berlangganan (**churn**). "
    "Silakan isi data pelanggan dan pilih model prediksi yang diinginkan.",
    icon="🔮"
)
st.markdown("---")


# --- Muat Aset ---
@st.cache_resource
def load_assets():
    base_path = 'saved_models'
    try:
        models = {
            "XGBoost (Seimbang)": joblib.load(os.path.join(base_path, 'xgboost_churn_model.pkl')),
            "Logistic Regression (Deteksi Maksimal)": joblib.load(os.path.join(base_path, 'logistic_regression_churn_model.pkl')),
            "Random Forest": joblib.load(os.path.join(base_path, 'random_forest_churn_model.pkl')),
            "Support Vector Machine": joblib.load(os.path.join(base_path, 'support_vector_machine_churn_model.pkl'))
        }
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        model_columns = joblib.load(os.path.join(base_path, 'model_columns.pkl'))
    except FileNotFoundError as e:
        st.error(f"Error memuat file model: {e}. Pastikan semua file .pkl ada di folder 'saved_models'.")
        return None, None, None
    return models, scaler, model_columns

models, scaler, model_columns = load_assets()

if models:
    # --- BUTTON 1: Info Model Toggle ---
    st.markdown("### 🎯 Pilih Model Prediksi")
    
    if 'show_model_info' not in st.session_state:
        st.session_state.show_model_info = False
    
    if st.button("ℹ️ Info Model" if not st.session_state.show_model_info else "❌ Tutup Info", 
                 type="secondary", use_container_width=True):
        st.session_state.show_model_info = not st.session_state.show_model_info
    
    if st.session_state.show_model_info:
        with st.container():
            st.subheader("📚 Penjelasan Model")
            
            # XGBoost
            with st.expander("🥇 XGBoost (Seimbang)", expanded=True):
                st.write("Model ensemble yang memberikan performa seimbang antara presisi dan recall. Cocok untuk prediksi umum dengan akurasi tinggi.")
                st.info("✅ **Keunggulan:** Akurasi tinggi, stabil, cepat dalam prediksi")
            
            # Logistic Regression
            with st.expander("🥈 Logistic Regression (Deteksi Maksimal)", expanded=True):
                st.write("Optimal untuk mendeteksi sebanyak mungkin pelanggan yang akan churn. Direkomendasikan untuk early warning system.")
                st.warning("🔍 **Keunggulan:** Deteksi maksimal churn, cocok untuk early warning")
            
            # Random Forest
            with st.expander("🥉 Random Forest", expanded=True):
                st.write("Model ensemble yang robust dengan interpretabilitas tinggi. Memberikan stabilitas prediksi yang baik.")
                st.success("🌳 **Keunggulan:** Interpretable, stabil, robust terhadap outlier")
            
            # SVM
            with st.expander("4️⃣ Support Vector Machine", expanded=True):
                st.write("Model yang efektif untuk klasifikasi dengan margin maksimal. Cocok untuk pattern recognition yang kompleks.")
                st.error("⚡ **Keunggulan:** Pattern recognition kompleks, margin optimal")
    
    # --- BUTTONS 2-5: Model Selection ---
    st.markdown("#### Pilih Model untuk Prediksi:")
    
    col_model1, col_model2 = st.columns(2)
    col_model3, col_model4 = st.columns(2)
    
    # Initialize session state for model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "XGBoost (Seimbang)"
    
    with col_model1:
        if st.button("🥇 XGBoost (Seimbang)", 
                    type="primary" if st.session_state.selected_model == "XGBoost (Seimbang)" else "secondary",
                    use_container_width=True):
            st.session_state.selected_model = "XGBoost (Seimbang)"
    
    with col_model2:
        if st.button("🥈 Logistic Regression", 
                    type="primary" if st.session_state.selected_model == "Logistic Regression (Deteksi Maksimal)" else "secondary",
                    use_container_width=True):
            st.session_state.selected_model = "Logistic Regression (Deteksi Maksimal)"
    
    with col_model3:
        if st.button("🥉 Random Forest", 
                    type="primary" if st.session_state.selected_model == "Random Forest" else "secondary",
                    use_container_width=True):
            st.session_state.selected_model = "Random Forest"
    
    with col_model4:
        if st.button("4️⃣ Support Vector Machine", 
                    type="primary" if st.session_state.selected_model == "Support Vector Machine" else "secondary",
                    use_container_width=True):
            st.session_state.selected_model = "Support Vector Machine"
    
    # Display selected model
    st.info(f"🎯 **Model Terpilih:** {st.session_state.selected_model}")
    
    # Dapatkan objek model yang dipilih
    chosen_model = models[st.session_state.selected_model]
    
    st.markdown("---")

    # --- BUTTONS 6-9: Quick Fill Options ---
    st.markdown("### 📋 Data Pelanggan")
    st.markdown("#### ⚡ Opsi Pengisian Cepat")
    
    col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
    
    with col_quick1:
        if st.button("👴 Pelanggan Senior", type="secondary", use_container_width=True):
            st.session_state.update({
                'senior_citizen': 1,
                'partner': 'Yes',
                'dependents': 'Yes',
                'tenure': 60,
                'contract': 'Two year',
                'monthly_charges': 70.0,
                'paperless_billing': 'No',
                'payment_method': 'Bank transfer (automatic)'
            })
            st.rerun()
    
    with col_quick2:
        if st.button("👨 Pelanggan Muda", type="secondary", use_container_width=True):
            st.session_state.update({
                'senior_citizen': 0,
                'partner': 'No',
                'dependents': 'No', 
                'tenure': 12,
                'contract': 'Month-to-month',
                'monthly_charges': 45.0,
                'paperless_billing': 'Yes',
                'payment_method': 'Electronic check'
            })
            st.rerun()
    
    with col_quick3:
        if st.button("💼 Pelanggan Bisnis", type="secondary", use_container_width=True):
            st.session_state.update({
                'internet_service': 'Fiber optic',
                'online_security': 'Yes',
                'tech_support': 'Yes',
                'contract': 'One year',
                'monthly_charges': 80.0,
                'streaming_tv': 'No',
                'streaming_movies': 'No'
            })
            st.rerun()
    
    with col_quick4:
        if st.button("🏠 Pelanggan Rumahan", type="secondary", use_container_width=True):
            st.session_state.update({
                'internet_service': 'DSL',
                'streaming_tv': 'Yes',
                'streaming_movies': 'Yes',
                'contract': 'Month-to-month',
                'monthly_charges': 50.0,
                'online_security': 'No',
                'tech_support': 'No'
            })
            st.rerun()
    
    # --- BUTTON 10: Reset Form ---
    col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 1])
    with col_reset2:
        if st.button("🔄 Reset Semua Form", type="secondary", use_container_width=True):
            # Clear all session state related to form
            keys_to_clear = ['senior_citizen', 'partner', 'dependents', 'tenure', 'contract', 
                           'internet_service', 'online_security', 'tech_support', 'streaming_tv', 
                           'streaming_movies', 'monthly_charges', 'paperless_billing', 'payment_method']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # --- Formulir Input ---
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 **Informasi Demografis**")
            SeniorCitizen = st.selectbox(
                'Lansia (Senior Citizen)', 
                [0, 1], 
                index=st.session_state.get('senior_citizen', 0),
                format_func=lambda x: '👴 Ya' if x == 1 else '👨 Tidak',
                help="Apakah pelanggan berusia 65 tahun atau lebih?"
            )
            Partner = st.selectbox(
                'Memiliki Partner', 
                ['Yes', 'No'],
                index=['Yes', 'No'].index(st.session_state.get('partner', 'No')),
                format_func=lambda x: '💑 Ya' if x == 'Yes' else '👤 Tidak',
                help="Apakah pelanggan memiliki pasangan?"
            )
            Dependents = st.selectbox(
                'Memiliki Tanggungan', 
                ['Yes', 'No'],
                index=['Yes', 'No'].index(st.session_state.get('dependents', 'No')),
                format_func=lambda x: '👨‍👩‍👧‍👦 Ya' if x == 'Yes' else '🚫 Tidak',
                help="Apakah pelanggan memiliki tanggungan keluarga?"
            )
            tenure = st.slider(
                'Lama Berlangganan (Bulan)', 
                min_value=0, 
                max_value=72, 
                value=st.session_state.get('tenure', 12),
                help="Berapa lama pelanggan telah berlangganan?"
            )
        
        with col2:
            st.markdown("#### 📞 **Layanan Telekomunikasi**")
            
            # BUTTONS 11-12: Service Level Quick Set
            col_svc1, col_svc2 = st.columns(2)
            with col_svc1:
                svc_full = st.form_submit_button("📶 Layanan Lengkap", type="secondary")
            with col_svc2:
                svc_min = st.form_submit_button("📱 Layanan Minimal", type="secondary")
            
            MultipleLines = st.selectbox(
                'Layanan Multiple Lines', 
                ['No', 'Yes', 'No phone service'],
                format_func=lambda x: '📞 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada layanan telepon'
            )
            InternetService = st.selectbox(
                'Layanan Internet', 
                ['DSL', 'Fiber optic', 'No'],
                index=['DSL', 'Fiber optic', 'No'].index(st.session_state.get('internet_service', 'DSL')),
                format_func=lambda x: '🌐 DSL' if x == 'DSL' else '⚡ Fiber Optic' if x == 'Fiber optic' else '🚫 Tidak ada'
            )
            OnlineSecurity = st.selectbox(
                'Layanan Online Security', 
                ['No', 'Yes', 'No internet service'],
                index=['No', 'Yes', 'No internet service'].index(st.session_state.get('online_security', 'No')),
                format_func=lambda x: '🔒 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
            OnlineBackup = st.selectbox(
                'Layanan Online Backup', 
                ['No', 'Yes', 'No internet service'],
                format_func=lambda x: '💾 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
            DeviceProtection = st.selectbox(
                'Layanan Device Protection', 
                ['No', 'Yes', 'No internet service'],
                format_func=lambda x: '🛡️ Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
            TechSupport = st.selectbox(
                'Layanan Tech Support', 
                ['No', 'Yes', 'No internet service'],
                index=['No', 'Yes', 'No internet service'].index(st.session_state.get('tech_support', 'No')),
                format_func=lambda x: '🔧 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
        
        with col3:
            st.markdown("#### 💰 **Layanan & Pembayaran**")
            
            # BUTTONS 13-14: Payment Type Quick Set
            col_pay1, col_pay2 = st.columns(2)
            with col_pay1:
                premium_btn = st.form_submit_button("💎 Premium", type="secondary")
            with col_pay2:
                basic_btn = st.form_submit_button("💡 Basic", type="secondary")
            
            StreamingTV = st.selectbox(
                'Layanan Streaming TV', 
                ['No', 'Yes', 'No internet service'],
                index=['No', 'Yes', 'No internet service'].index(st.session_state.get('streaming_tv', 'No')),
                format_func=lambda x: '📺 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
            StreamingMovies = st.selectbox(
                'Layanan Streaming Movies', 
                ['No', 'Yes', 'No internet service'],
                index=['No', 'Yes', 'No internet service'].index(st.session_state.get('streaming_movies', 'No')),
                format_func=lambda x: '🎬 Ya' if x == 'Yes' else '🚫 Tidak' if x == 'No' else '❌ Tidak ada internet'
            )
            Contract = st.selectbox(
                'Jenis Kontrak', 
                ['Month-to-month', 'One year', 'Two year'],
                index=['Month-to-month', 'One year', 'Two year'].index(st.session_state.get('contract', 'Month-to-month')),
                format_func=lambda x: '📅 Bulanan' if x == 'Month-to-month' else '📆 1 Tahun' if x == 'One year' else '🗓️ 2 Tahun'
            )
            PaperlessBilling = st.selectbox(
                'Tagihan Elektronik', 
                ['Yes', 'No'],
                index=['Yes', 'No'].index(st.session_state.get('paperless_billing', 'Yes')),
                format_func=lambda x: '📧 Ya' if x == 'Yes' else '📄 Tidak'
            )
            PaymentMethod = st.selectbox(
                'Metode Pembayaran', 
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(st.session_state.get('payment_method', 'Electronic check')),
                format_func=lambda x: {
                    'Electronic check': '💳 Cek Elektronik',
                    'Mailed check': '📮 Cek Pos',
                    'Bank transfer (automatic)': '🏦 Transfer Bank (Otomatis)',
                    'Credit card (automatic)': '💳 Kartu Kredit (Otomatis)'
                }[x]
            )
            MonthlyCharges = st.number_input(
                'Biaya Bulanan ($)', 
                min_value=0.0, 
                max_value=120.0, 
                value=float(st.session_state.get('monthly_charges', 50.0)),
                step=0.5,
                help="Biaya yang dibayar pelanggan setiap bulan"
            )
            TotalCharges = st.number_input(
                'Total Biaya ($)', 
                min_value=0.0, 
                max_value=9000.0, 
                value=500.0,
                step=10.0,
                help="Total biaya yang telah dibayar pelanggan"
            )
        
        # Handle quick service buttons
        if svc_full:
            st.session_state.update({
                'online_security': 'Yes',
                'online_backup': 'Yes',
                'device_protection': 'Yes',
                'tech_support': 'Yes',
                'streaming_tv': 'Yes',
                'streaming_movies': 'Yes'
            })
            st.rerun()
        
        if svc_min:
            st.session_state.update({
                'online_security': 'No',
                'online_backup': 'No',
                'device_protection': 'No',
                'tech_support': 'No',
                'streaming_tv': 'No',
                'streaming_movies': 'No'
            })
            st.rerun()
        
        # Handle payment type buttons
        if premium_btn:
            st.session_state.update({
                'monthly_charges': 85.0,
                'contract': 'One year',
                'payment_method': 'Credit card (automatic)',
                'paperless_billing': 'Yes'
            })
            st.rerun()
        
        if basic_btn:
            st.session_state.update({
                'monthly_charges': 35.0,
                'contract': 'Month-to-month',
                'payment_method': 'Electronic check',
                'paperless_billing': 'No'
            })
            st.rerun()
        
        # BUTTON 15: Main Prediction Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            submitted = st.form_submit_button(
                "🔮 Prediksi Churn", 
                use_container_width=True,
                type="primary"
            )

    # --- Logika Prediksi ---
    # --- (Salin dan GANTI seluruh blok if submitted di kode Anda dengan ini) ---

if submitted:
    with st.spinner('🔄 Sedang memproses prediksi...'):
        # Sesuai dengan pipeline di Colab Anda:
        input_dict = {
            'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents, 'tenure': tenure,
            'MultipleLines': MultipleLines, 'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
            'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies, 'Contract': Contract,
            'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }
        input_df = pd.DataFrame([input_dict])

        # Menangani kasus khusus: jika tenure=0, maka TotalCharges harus 0
        if input_df['tenure'].iloc[0] == 0:
            input_df['TotalCharges'] = 0.0

        # Pra-pemrosesan persis seperti di Colab
        kolom_diubah = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for kolom in kolom_diubah:
            input_df[kolom] = input_df[kolom].replace("No internet service", "No")
        input_df["MultipleLines"] = input_df["MultipleLines"].replace("No phone service", "No")

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_aligned)
        
        # Prediksi menggunakan model yang dipilih
        prediction = chosen_model.predict(input_scaled)
        prediction_proba = chosen_model.predict_proba(input_scaled)
    
    # --- Tampilkan Hasil dengan Styling ---
    # Pastikan prediksi berhasil sebelum menampilkan hasil
    if prediction_proba is not None and len(prediction_proba) > 0:
        st.markdown("---")
        st.markdown(f"### 📊 Hasil Prediksi: **{st.session_state.selected_model}**")
        
        # Hasil utama dengan layout yang menarik
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ **RISIKO TINGGI**")
                st.write("**Pelanggan diprediksi akan CHURN**")
                st.caption("Disarankan untuk mengambil tindakan retensi")
            else:
                st.success("✅ **AMAN**")
                st.write("**Pelanggan diprediksi akan SETIA**")
                st.caption("Pelanggan kemungkinan akan melanjutkan layanan")
        
        with col2:
            st.write("")  # Empty space for layout
        
        with col3:
            confidence = prediction_proba.max() * 100
            if confidence >= 80:
                status = "Sangat Tinggi"
                st.success(f"**Tingkat Kepercayaan**")
            elif confidence >= 60:
                status = "Tinggi"
                st.warning(f"**Tingkat Kepercayaan**")
            else:
                status = "Rendah"
                st.error(f"**Tingkat Kepercayaan**")
            
            st.metric("Confidence Level", f"{confidence:.1f}%", status)

        # Detail probabilitas
        st.markdown("#### 📈 Detail Probabilitas")
        col1, col2 = st.columns(2)

        churn_prob = prediction_proba[0][1] * 100
        no_churn_prob = prediction_proba[0][0] * 100

        with col1:
            st.metric(
                label="🔴 Probabilitas Churn", 
                value=f"{churn_prob:.2f}%",
                delta=f"{churn_prob - 50:.1f}%" if churn_prob > 50 else None
            )

        with col2:
            st.metric(
                label="🟢 Probabilitas Tidak Churn", 
                value=f"{no_churn_prob:.2f}%",
                delta=f"{no_churn_prob - 50:.1f}%" if no_churn_prob > 50 else None
            )

        # BAGIAN VISUALISASI PROBABILITAS TELAH DIHAPUS
        
        # Rekomendasi berdasarkan hasil
        st.markdown("#### 💡 Rekomendasi Tindakan")
        if prediction[0] == 1:
            if churn_prob > 80:
                st.warning("""
                **Tindakan Segera Diperlukan:**
                - 📞 Hubungi pelanggan dalam 24 jam
                - 🎁 Tawarkan promosi khusus atau diskon
                - 🤝 Jadwalkan konsultasi untuk memahami kebutuhan
                - 📊 Review layanan yang saat ini digunakan
                """)
            else:
                st.info("""
                **Tindakan Preventif:**
                - 📧 Kirim email dengan penawaran menarik
                - 📋 Survey kepuasan pelanggan
                - 🆙 Upgrade layanan dengan benefit tambahan
                """)
        else:
            st.success("""
            **Strategi Retention:**
            - 🌟 Pelanggan loyal - pertahankan layanan berkualitas
            - 📈 Tawarkan upgrade layanan untuk meningkatkan value
            - 🎯 Jadikan referral untuk mendapat pelanggan baru
            """)
        
        # --- BUTTONS 16-19: Action Buttons setelah prediksi ---
        st.markdown("#### 🎯 Tindakan Selanjutnya")
        
        col_action1, col_action2, col_action3, col_action4 = st.columns(4)
        
        with col_action1:
            # BUTTON 16: Export Results
            # Dibuat agar download button muncul setelah di-klik, bukan otomatis
            export_data = {
                    'Model': st.session_state.selected_model,
                    'Prediksi': 'CHURN' if prediction[0] == 1 else 'TIDAK CHURN',
                    'Confidence': f"{prediction_proba.max()*100:.2f}%",
                    'Prob_Churn': f"{churn_prob:.2f}%",
                    'Prob_NoChurn': f"{no_churn_prob:.2f}%",
                    'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents,
                    'Tenure': tenure, 'Contract': Contract, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
            }
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📄 Export Hasil",
                data=csv,
                file_name=f"prediksi_churn_{st.session_state.selected_model.replace(' ', '_')}.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )
        
        with col_action2:
            # BUTTON 17: Generate Email Template
            if st.button("📧 Generate Email", type="secondary", use_container_width=True):
                if prediction[0] == 1:
                    email_template = f"""Subject: Urgent: Customer Retention Required

Dear Customer Service Team,
Our AI model ({st.session_state.selected_model}) has identified a high-risk customer:
- Risk Level: {confidence:.1f}%
- Churn Probability: {churn_prob:.1f}%
- Customer Profile: 
 * Tenure: {tenure} months
 * Monthly Charges: ${MonthlyCharges}
 * Contract: {Contract}

Recommended Actions:
{'- Priority contact within 24 hours' if churn_prob > 80 else '- Proactive outreach recommended'}
- Offer retention incentives
- Review service satisfaction

Please take immediate action.
Best regards,
AI Prediction System"""
                else:
                    email_template = f"""Subject: Customer Loyalty Opportunity

Dear Account Manager,
Great news! Our AI model shows this customer is likely to stay:
- Loyalty Score: {confidence:.1f}%
- Customer Profile: 
 * Tenure: {tenure} months
 * Monthly Charges: ${MonthlyCharges}

Recommended Actions:
- Consider upselling opportunities
- Request referrals
- Maintain excellent service

Best regards,
AI Prediction System"""
                
                st.text_area("📧 Email Template Generated:", email_template, height=300)
        
        with col_action3:
            # BUTTON 18: Re-analyze with All Models
            if st.button("📊 Analisis Ulang", type="secondary", use_container_width=True):
                st.info("🔄 Menjalankan prediksi dengan model lain untuk perbandingan...")
                
                comparison_results = {}
                for model_name, model in models.items():
                    pred = model.predict(input_scaled)
                    pred_proba = model.predict_proba(input_scaled)
                    comparison_results[model_name] = {
                        'Prediksi': 'CHURN' if pred[0] == 1 else 'TIDAK CHURN',
                        'Confidence': f"{pred_proba.max()*100:.2f}%",
                        'Churn_Prob': f"{pred_proba[0][1]*100:.2f}%"
                    }
                
                comparison_df = pd.DataFrame(comparison_results).T
                st.dataframe(comparison_df, use_container_width=True)
        
        with col_action4:
            # BUTTON 19: New Prediction
            if st.button("🔄 Prediksi Baru", type="secondary", use_container_width=True):
                keys_to_clear = ['senior_citizen', 'partner', 'dependents', 'tenure', 'contract', 
                                 'internet_service', 'online_security', 'tech_support', 'streaming_tv', 
                                 'streaming_movies', 'monthly_charges', 'paperless_billing', 'payment_method']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    else:
        # Menampilkan pesan jika prediksi gagal
        st.error("Gagal melakukan prediksi. Model tidak memberikan output yang valid.")
        st.info("Hal ini bisa terjadi jika kombinasi input sangat tidak biasa. Coba ubah input Anda dan lakukan prediksi lagi.")

# --- FOOTER (Selaraskan dengan Page 2) ---
st.markdown("---")
st.markdown("<center><span style='color: #999;'>© 2025 Kelompok 2 Data Mining</span></center>", unsafe_allow_html=True)
