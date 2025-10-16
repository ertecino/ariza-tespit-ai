# streamlit_app.py

import streamlit as st
import pandas as pd
import ariza_ai

@st.cache_data
def get_master_dataframe():
    """
    Tüm uygulama için gerekli olan, zenginleştirilmiş ana veri çerçevesini oluşturur.
    """
    # DÜZELTME: Dosya adını doğru .xlsx adıyla güncelledik.
    df_raw = ariza_ai.load_data('Operasyonel Verimsizlik (İstasyon Arızaları) İçin Veri Seti.xlsx')
    if df_raw is None:
        st.error("Veri dosyası yüklenemedi. Lütfen dosya adının ve konumunun doğru olduğundan emin olun.")
        st.stop()
        
    df_processed_for_training = ariza_ai.preprocess_data(df_raw.copy())
    model, columns, accuracy = ariza_ai.train_and_evaluate_model(df_processed_for_training)
    
    master_df = df_raw.copy()
    processed_for_prediction = ariza_ai.preprocess_data(master_df.copy())
    
    # 'ariza_oldu_mu' sütunu varsa çıkar, yoksa hata vermeden devam et
    if 'ariza_oldu_mu' in processed_for_prediction.columns:
        X_full = processed_for_prediction.drop('ariza_oldu_mu', axis=1)
    else:
        X_full = processed_for_prediction

    X_full = X_full.reindex(columns=columns, fill_value=0)
    all_risk_scores = model.predict_proba(X_full)[:, 1]
    
    master_df['Arıza Riski'] = all_risk_scores
    
    return master_df, model, columns, accuracy

st.set_page_config(layout="wide")
st.title('⚡ ArızaTespit AI - Operasyonel Zeka Dashboard')

master_data, model, model_columns, accuracy = get_master_dataframe()

st.sidebar.title("Proje Bilgileri")
st.sidebar.info("Bu prototip, bir istasyonun arıza riskini tahmin ederek proaktif bakım yapılmasını sağlayan bir karar destek sistemidir.")
st.sidebar.subheader("Model Performansı")
st.sidebar.metric(label="Test Doğruluk Oranı", value=f"{accuracy:.2%}")
st.sidebar.write("Kullanılan Model: Lojistik Regresyon")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Detaylı Analiz", "⚙️ Risk Simülasyonu"])

with tab1:
    st.header("Anlık Operasyon Durumu")
    st.subheader("Genel Bakış")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kayıt Sayısı", len(master_data))
    arizali_sayisi = master_data['ariza_oldu_mu'].sum()
    col2.metric("Tespit Edilen Arıza Sayısı", arizali_sayisi)
    col3.metric("Genel Arıza Oranı", f"{arizali_sayisi/len(master_data):.2%}")
    
    st.write("---")
    
    st.subheader("🚨 Yüksek Riskli İstasyonlar (Öncelikli Aksiyon Listesi)")
    st.dataframe(
        master_data.nlargest(5, 'Arıza Riski')[['istasyon_id', 'son_bakimdan_gecen_gun', 'voltaj_dalgalanmasi_sayisi', 'Arıza Riski']],
        use_container_width=True,
        column_config={
            "Arıza Riski": st.column_config.ProgressColumn("Arıza Riski", format="%.2f", min_value=0, max_value=1),
            "son_bakimdan_gecen_gun": st.column_config.NumberColumn("Bakım Gecikmesi (Gün)")
        }
    )

with tab2:
    st.header("Veri Odaklı İçgörüler")
    st.subheader("🔍 Arıza Kök Neden Analizi")
    st.write("Bakım Gecikmesi ve Voltaj Dalgalanmalarının Arıza Durumuna Etkisi")
    
    master_data['Arıza Etiketi'] = master_data['ariza_oldu_mu'].apply(lambda x: 'Arızalı' if x == 1 else 'Sağlam')
    st.scatter_chart(
        master_data,
        x='son_bakimdan_gecen_gun',
        y='voltaj_dalgalanmasi_sayisi',
        color='Arıza Etiketi',
        size='gunluk_sarj_sayisi'
    )
    st.info("Grafik Yorumu: Genellikle bakımı gecikmiş (sağ eksen) ve voltaj dalgalanması yaşamış (üst eksen) istasyonlarda arıza ('Arızalı' noktalar) yoğunlaşmaktadır.")

    st.write("---")
    
    st.subheader("📖 Tüm Veri Setinin Detaylı Durumu")
    st.dataframe(master_data, use_container_width=True)

with tab3:
    st.header("Tekil İstasyon Senaryo Analizi")
    st.info("💡 Farklı senaryoları test etmek için aşağıdaki formu kullanın.")
    
    with st.form("simulation_form"):
        st.write("**Bir istasyonun arıza riskini tahmin etmek için aşağıdaki bilgileri girin:**")
        c1, c2 = st.columns(2)
        with c1:
            usage = st.slider('Günlük Şarj Sayısı', 1, 100, 20)
            duration = st.slider('Ortalama Şarj Süresi (dk)', 10, 240, 45)
            voltage_spikes = st.slider('Voltaj Dalgalanması Sayısı', 0, 50, 5)
        with c2:
            days_since_maintenance = st.number_input('Son Bakımdan Beri Geçen Gün Sayısı', min_value=0, max_value=2000, value=365)
            error_code_options = ['Yok'] + list(master_data['hata_kodu'].unique())
            error_code = st.selectbox('Son Alınan Hata Kodu', error_code_options)
        
        submitted = st.form_submit_button("Riski Hesapla")

        if submitted:
            user_inputs = {
                'gunluk_sarj_sayisi': usage,
                'ortalama_sarj_suresi_dk': duration,
                'voltaj_dalgalanmasi_sayisi': voltage_spikes,
                'son_bakimdan_gecen_gun': days_since_maintenance,
                'hata_kodu': error_code if error_code != 'Yok' else None
            }
            risk_score = ariza_ai.predict_failure_risk(model, model_columns, user_inputs)
            
            st.subheader("📊 Tahmin Sonucu")
            col1, col2 = st.columns(2)
            col1.metric("Hesaplanan Arıza Riski", f"{risk_score:.0%}")
            if risk_score > 0.75: col2.error("DURUM: YÜKSEK RİSK!")
            elif risk_score > 0.5: col2.warning("DURUM: ORTA RİSK.")
            else: col2.success("DURUM: DÜŞÜK RİSK.")
            st.progress(int(risk_score * 100))
            
            st.subheader("💡 Analiz Özeti")
            if risk_score > 0.5:
                st.info(f"Model analizine göre, bu istasyonun riskini artıran ana faktörler:\n- **Bakım Gecikmesi:** {days_since_maintenance} gün\n- **Voltaj Dalgalanmaları:** {voltage_spikes} adet")
            else:
                st.info("İstasyonun operasyonel değerleri normal sınırlar içinde.")