# ariza_ai.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(filepath: str) -> pd.DataFrame:
    """
    Verilen yoldan arıza veri setini .xlsx formatında yükler.
    """
    print("1. Excel veri seti yükleniyor...")
    try:
        # DÜZELTME: CSV yerine Excel okuyucu kullanılıyor.
        return pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"HATA: Belirtilen yolda dosya bulunamadı: {filepath}")
        return None
    except Exception as e:
        print(f"HATA: Dosya okunurken bir sorun oluştu: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setini modelin anlayacağı formata dönüştürür.
    Yeni veri yapısına göre güncellenmiştir.
    """
    print("2. Veri, model için hazırlanıyor (Ön İşleme)...")
    
    # 'hata_kodu' sütununu sayısallaştır (One-Hot Encoding)
    if 'hata_kodu' in df.columns:
        df = pd.get_dummies(df, columns=['hata_kodu'], drop_first=True, dtype=int)
    
    # Model için gereksiz olan, tanımlayıcı sütunları kaldır
    columns_to_drop = ['istasyon_id', 'tarih']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df

def train_and_evaluate_model(df: pd.DataFrame):
    """
    Lojistik Regresyon modelini eğitir ve performansını değerlendirir.
    Yeni hedef sütun adına göre güncellenmiştir.
    """
    print("3. Yapay Zeka Modeli eğitiliyor ve test ediliyor...")
    
    target_column = 'ariza_oldu_mu'
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' sütunu veri setinde bulunamadı.")
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Veri setini eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modeli oluştur ve eğit
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Model performansını test et
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   >>> Modelin Test Başarısı (Doğruluk Oranı): {accuracy:.2f} ({accuracy*100:.0f}%)")
    
    return model, X.columns, accuracy

def predict_failure_risk(model, columns, station_data: dict) -> float:
    """
    Verilen yeni bir istasyon verisi için arıza riskini % olarak tahmin eder.
    Yeni veri yapısına göre güncellenmiştir.
    """
    input_df = pd.DataFrame([station_data])
    
    if 'hata_kodu' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['hata_kodu'], dtype=int)
        
    # Eğitimdeki sütunlarla tam uyumlu hale getir (eksik sütunları 0 ile doldur)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Arıza olasılığını (risk skorunu) hesapla
    failure_probability = model.predict_proba(input_df)[:, 1]
    
    return failure_probability[0]
    