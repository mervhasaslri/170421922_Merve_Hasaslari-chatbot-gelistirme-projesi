import requests

API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_llama_response(prompt, pdf_content):
    system_prompt = (
        "Sen Prizren şehri hakkında bilgi veren bir asistansın. "
        "Aşağıdaki bilgileri kullanarak soruları yanıtla:\n\n"
        f"{pdf_content}\n\n"
        "Yanıtını Türkçe olarak ver."
    )
    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Llama API hatası: {response.status_code} - {response.text}" 