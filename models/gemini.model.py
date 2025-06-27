import google.generativeai as genai

genai.configure(api_key="AIzaSyCIO2lIc5u2F7_7lOSQFqwb295wfMr51-c")
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def get_gemini_response(prompt, pdf_content):
    full_prompt = f"""
    Sen Prizren şehri hakkında bilgi veren bir asistansın. 
    Aşağıdaki bilgileri kullanarak soruları yanıtla:
    
    {pdf_content}
    
    Kullanıcı sorusu: {prompt}
    
    Lütfen yanıtını Türkçe olarak ver.
    """
    response = model.generate_content(full_prompt)
    return response.text 
