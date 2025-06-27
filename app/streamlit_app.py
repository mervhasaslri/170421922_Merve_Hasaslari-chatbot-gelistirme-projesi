import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from models.gemini_model import get_gemini_response
from models.llama_model import get_llama_response
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Prizren Chatbot",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #f7f7f9;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .chat-bubble-user {background: #e0f7fa; color: #006064; border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; margin-right: 30%; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    .chat-bubble-assistant {background: #fff3e0; color: #e65100; border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; margin-left: 30%; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
    .stChatInput input {font-size: 1.1rem; padding: 0.75rem;}
    .sidebar-info {background: #fff; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem;}
    .sidebar-img {width: 100%; border-radius: 8px; margin-bottom: 1rem;}
    .app-title {font-size: 2.5rem; font-weight: bold; color: #2d2d2d; margin-bottom: 0.5rem;}
    .app-desc {font-size: 1.2rem; color: #555; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

def read_pdf():
    pdf_path = "data/prizren_bilgileri.pdf"
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"PDF okuma hatası: {str(e)}")
    return text

def read_example_questions():
    try:
        df = pd.read_csv("data/chatbot_dataset_prizren_updated.csv")
        questions = df['Soru'].tolist() if 'Soru' in df.columns else []
        if not questions:
            questions = [
                "Prizren'in tarihi hakkında bilgi verir misin?",
                "Prizren'de gezilecek yerler nerelerdir?",
                "Prizren'in meşhur yemekleri nelerdir?",
                "Prizren'de hangi dillerde konuşulur?",
                "Prizren'in nüfusu ne kadar?"
            ]
        return questions
    except Exception as e:
        st.error(f"CSV okuma hatası: {str(e)}")
        return [
            "Prizren'in tarihi hakkında bilgi verir misin?",
            "Prizren'de gezilecek yerler nerelerdir?",
            "Prizren'in meşhur yemekleri nelerdir?",
            "Prizren'de hangi festivaller düzenlenir?",
            "Prizren'in nüfusu ne kadar?"
        ]

# Embedding tabanlı cevaplayıcı fonksiyon
def get_embedding_response(prompt, df, pdf_content, model_option):
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    questions = df['Soru'].astype(str).tolist() if 'Soru' in df.columns else df[df.columns[0]].astype(str).tolist()
    all_embeddings = embedder.encode(questions)
    prompt_emb = embedder.encode([prompt])[0]
    sims = cosine_similarity([prompt_emb], all_embeddings)[0]
    idx = sims.argmax()
    closest_question = questions[idx]
    # En yakın soruyu seçili modele gönder
    if model_option == "Gemini":
        return get_gemini_response(closest_question, pdf_content)
    elif model_option == "Llama 3.3 8B Instruct":
        return get_llama_response(closest_question, pdf_content)
    
    else:
        return "Bir hata oluştu."

def main():
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.image("prizren.jpg", caption="Prizren, Kosova", use_column_width=True)
        st.markdown("**Prizren**, Kosova'nın en güzel ve tarihi şehirlerinden biridir. Osmanlı'dan kalma taş köprüleri, camileri ve dar sokaklarıyla ünlüdür. Kültürel çeşitliliği ve doğal güzellikleriyle ziyaretçilerini büyüler. Prizren'in tarihi ve kültürel değerleri, UNESCO'nun dünya mirası listesine dahil edilmiştir. Masallardan çıkmış bu şehire sizi bekleriz...")
        st.markdown('</div>', unsafe_allow_html=True)
        # Model seçimi
        model_option = st.selectbox("Kullanmak istediğiniz modeli seçin:", ["Gemini", "Llama 3.3 8B Instruct", "Embedding"])
        example_questions = read_example_questions()
        if example_questions:
            st.markdown("### Örnek Sorular")
            for question in example_questions[:5]:
                if st.button(question, key=question):
                    prompt = question
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    pdf_content = read_pdf()
                    if model_option == "Gemini":
                        response_text = get_gemini_response(prompt, pdf_content)
                    elif model_option == "Llama 3.3 8B Instruct":
                        response_text = get_llama_response(prompt, pdf_content)
                    else:
                        df = pd.read_csv("data/chatbot_dataset_prizren_updated.csv")
                        response_text = get_embedding_response(prompt, df, pdf_content, "Gemini")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.experimental_rerun()

    st.markdown('<div class="app-title">🏛️ Prizren Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-desc">Prizren hakkında bilgi almak için sorularınızı aşağıya yazabilirsiniz!</div>', unsafe_allow_html=True)

    pdf_content = read_pdf()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Prizren hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Düşünüyorum..."):
            if 'model_option' in locals():
                selected_model = model_option
            else:
                selected_model = "Gemini"
            if selected_model == "Gemini":
                response_text = get_gemini_response(prompt, pdf_content)
            elif selected_model == "Llama 3.3 8B Instruct":
                response_text = get_llama_response(prompt, pdf_content)
            else:
                df = pd.read_csv("data/chatbot_dataset_prizren_updated.csv")
                response_text = get_embedding_response(prompt, df, pdf_content, "Gemini")
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.experimental_rerun()

if __name__ == "__main__":
    main() 
