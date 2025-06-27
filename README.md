# 170421922_Merve_Hasaslari-chatbot-gelistirme-projesi
Bu proje, seçilen bir konuda, yapay zekâ destekli bir chatbot geliştirme sürecini kapsamaktadır. Prizren hakkında bilgi edinmeye yarayan bir chatbot'tur.

# Prizren Chatbot

Bu proje, Prizren hakkında bilgi veren çok modelli bir sohbet botudur. Kullanıcılar Gemini, Llama 3.3 8B Instruct veya embedding tabanlı arama ile sohbet edebilir. Ayrıca, modellerin performansını karşılaştırmak için bir değerlendirme scripti de içerir.

## Özellikler
- **Çoklu Model Desteği:** Gemini, Llama 3.3 8B Instruct (OpenRouter API) ve Embedding tabanlı cevaplama
- **PDF'den Bilgi:** Prizren ile ilgili bilgiler PDF dosyasından alınır
- **Örnek Sorular:** Sidebar'da örnek sorular
- **Türkçe Sohbet:** Yanıtlar Türkçe döner
- **Model Karşılaştırma:** Precision, Recall, F1 Score ve Confusion Matrix ile model performans karşılaştırması

## Kurulum
1. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers scikit-learn tabulate openai
   ```
2. `data/` klasöründe şu dosyaların olduğundan emin olun:
   - `chatbot_dataset_prizren_updated.csv` (Sütunlar: `Intent`, `Örnek Cümle`)
   - `prizren_bilgileri.pdf`
3. API anahtarlarınızı ilgili model dosyalarına ekleyin (ör. `models/llama_model.py` içinde doğrudan kodda tanımlı).

## Kullanım
### Chatbot Arayüzü
Uygulamayı başlatmak için:
```bash
streamlit run app/streamlit_app.py
```
- Sidebar'dan model seçin: **Gemini**, **Llama 3.3 8B Instruct** veya **Embedding**
- Sorunuzu yazın veya örnek sorulardan birine tıklayın

### Model Karşılaştırma
Farklı modellerin aynı veri üzerinde başarımını karşılaştırmak için:
```bash
python compare_models.py
```
- Sonuçlar terminalde tablo olarak gösterilir
- Precision, Recall, F1 Score ve Confusion Matrix değerleri yazdırılır

## Dosya Yapısı
```
chatbot_prizren/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── gemini_model.py
│   └── llama_model.py
├── data/
│   ├── chatbot_dataset_prizren_updated.csv
│   └── prizren_bilgileri.pdf
├── compare_models.py
├── train_embedding_model.py
├── requirements.txt
└── README.md
```

## Notlar
- Embedding seçeneğinde, kullanıcının sorusuna en yakın örnek cümle embedding ile bulunur ve seçili modelle cevaplanır.
- CSV dosyanızda sütun adları **Intent** ve **Örnek Cümle** olmalıdır.
- Llama ve Gemini için API anahtarlarınızı kodda güncellediğinizden emin olun.

Her türlü soru ve katkı için iletişime geçebilirsiniz! 

