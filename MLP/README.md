# Yapay Sinir Ağları ile Banknot Doğrulama

Bu proje, banknotların özelliklerine dayanarak gerçek mi yoksa sahte mi olduğunu sınıflandırmak için çeşitli makine öğrenimi modellerini, özel olarak geliştirilmiş sinir ağları ile PyTorch/Scikit-learn uygulamalarını karşılaştırmaktadır.

## Veri Seti

Bu projede kullanılan veri seti `BankNote_Authentication.csv` olup aşağıdaki özellikleri içermektedir:

- **Varyans**: Dalgacık dönüşümü yapılmış görüntünün varyansı.
- **Çarpıklık**: Dalgacık dönüşümü yapılmış görüntünün çarpıklığı.
- **Basıklık**: Dalgacık dönüşümü yapılmış görüntünün basıklığı.
- **Entropi**: Görüntünün entropisi.
- **Sınıf**: Hedef değişken (0 gerçek banknot, 1 sahte banknot).

Veri seti, %80-20 oranında eğitim ve test setlerine ayrılmıştır.

---

## Uygulanan Modeller

### 1. **Özel 2 Katmanlı Sinir Ağı**
- `mlp_2layer.py` içinde uygulanmıştır.
- Mimari:
  - Giriş katmanı: Veri setindeki özellik sayısı.
  - Gizli katman: Yapılandırılabilir nöron sayısı (`n_h`).
  - Çıkış katmanı: Sigmoid aktivasyon fonksiyonlu tek nöron.
- Aktivasyon fonksiyonları:
  - Gizli katman: ReLU veya Tanh (yapılandırılabilir).
  - Çıkış katmanı: Sigmoid.
- Optimizasyon algoritması: Gradyan İniş (Gradient Descent).
- Kayıp fonksiyonu: İkili Çapraz Entropi (Binary Cross-Entropy).

### 2. **Özel 3 Katmanlı Sinir Ağı**
- `mlp_3layer.py` içinde uygulanmıştır.
- Mimari:
  - Giriş katmanı: Veri setindeki özellik sayısı.
  - İki gizli katman: Yapılandırılabilir nöron sayısı (`n_h1` ve `n_h2`).
  - Çıkış katmanı: Sigmoid aktivasyon fonksiyonlu tek nöron.
- Aktivasyon fonksiyonları:
  - Gizli katmanlar: Tanh.
  - Çıkış katmanı: Sigmoid.
- Optimizasyon algoritması: Gradyan İniş (Gradient Descent).
- Kayıp fonksiyonu: İkili Çapraz Entropi (Binary Cross-Entropy).

### 3. **Scikit-learn MLPClassifier**
- Scikit-learn'ün `MLPClassifier`'ını aşağıdaki yapılandırma ile kullanır:
  - Gizli katmanlar: Her biri `n_hidden` nöronlu iki katman.
  - Aktivasyon fonksiyonu: ReLU.
  - Optimizasyon algoritması: Stokastik Gradyan İniş (SGD).
  - Kayıp fonksiyonu: İkili Çapraz Entropi (Binary Cross-Entropy).

### 4. **PyTorch Sinir Ağı**
- PyTorch kullanılarak `train3.py` içinde uygulanmıştır.
- Mimari:
  - Giriş katmanı: Veri setindeki özellik sayısı.
  - İki gizli katman: Yapılandırılabilir nöron sayısı (`n_hidden`).
  - Çıkış katmanı: Sigmoid aktivasyon fonksiyonlu tek nöron.
- Aktivasyon fonksiyonları:
  - Gizli katmanlar: ReLU.
  - Çıkış katmanı: Sigmoid.
- Optimizasyon algoritması: Stokastik Gradyan İniş (SGD).
- Kayıp fonksiyonu: İkili Çapraz Entropi (Binary Cross-Entropy).

---

## Değerlendirme Metrikleri

Modeller aşağıdaki metrikler kullanılarak değerlendirilir:
- **Doğruluk (Accuracy)**: Doğru sınıflandırılan örneklerin yüzdesi.
- **Kesinlik (Precision)**: Gerçek pozitiflerin, tüm tahmin edilen pozitiflere oranı.
- **Geri Çağırma (Recall)**: Gerçek pozitiflerin, tüm gerçek pozitiflere oranı.
- **F1 Skoru**: Kesinlik ve geri çağırmanın harmonik ortalaması.
- **Karmaşıklık Matrisi (Confusion Matrix)**: Gerçek pozitiflerin, gerçek negatiflerin, yanlış pozitiflerin ve yanlış negatiflerin görsel temsili.
- **ROC-AUC Skoru**: ROC eğrisi altındaki alan (olasılıklar mevcutsa).

---

## Sonuçların Görselleştirilmesi

Aşağıdaki görselleştirmeler oluşturulur:
1. **Karmaşıklık Matrisi**: Her model için karmaşıklık matrislerini çizer.
2. **Öğrenme Eğrisi**: Özel modeller için iterasyonlar boyunca eğitim ve doğrulama kaybını çizer.

---

## Nasıl Çalıştırılır

### Ön Koşullar
- Python 3.8 veya daha yüksek sürüm.
- Gerekli kütüphaneler:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `torch` (PyTorch)

Gerekli kütüphaneleri şu komutla yükleyin:
```bash
pip install -r requirements.txt
```
veya doğrudan:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch
```