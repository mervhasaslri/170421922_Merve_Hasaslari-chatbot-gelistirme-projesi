import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tabulate import tabulate
from models.gemini_model import get_gemini_response
from models.llama_model import get_llama_response

# 1. Veri Yükleme
csv_path = "data/chatbot_dataset_prizren_updated.csv"
df = pd.read_csv(csv_path)

# 2. Soru ve etiket sütunlarını seç
X = df['Örnek Cümle'].astype(str)
y = df['Intent']

# 3. Train/Test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. PDF içeriğini oku (cevaplar için gerekli)
def read_pdf():
    pdf_path = "data/prizren_bilgileri.pdf"
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"PDF okuma hatası: {str(e)}")
    return text

pdf_content = read_pdf()

# 5. Her model için tahminler
models = {
    "Gemini": get_gemini_response,
    "Llama": get_llama_response
}

results = {}

for model_name, model_func in models.items():
    y_pred = []
    print(f"{model_name} ile tahminler yapılıyor...")
    for question in X_test:
        try:
            pred = model_func(question, pdf_content)
        except Exception as e:
            pred = ""
        y_pred.append(pred)
    # Burada, modelin döndürdüğü cevabı doğrudan etiketle karşılaştırmak yerine, bir mapping veya string benzerliği ile eşleştirme yapılabilir.
    # Basitlik için, cevap tam eşleşiyorsa doğru kabul edilecek:
    y_pred_label = [p if p in y_test.values else "other" for p in y_pred]
    results[model_name] = {
        "precision": precision_score(y_test, y_pred_label, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred_label, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred_label, average='weighted', zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred_label, labels=list(set(y_test)))
    }

# 6. Sonuçları tablo olarak yazdır
print("\nModel Performans Karşılaştırması:")
table = []
for model_name, metrics in results.items():
    table.append([
        model_name,
        f"{metrics['precision']:.2f}",
        f"{metrics['recall']:.2f}",
        f"{metrics['f1']:.2f}"
    ])
print(tabulate(table, headers=["Model", "Precision", "Recall", "F1 Score"], tablefmt="github"))

# (İsteğe bağlı) Confusion Matrix yazdır
for model_name, metrics in results.items():
    print(f"\n{model_name} Confusion Matrix:")
    print(metrics['confusion']) 