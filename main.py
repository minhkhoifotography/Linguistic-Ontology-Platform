from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="VnQuill Pro AI Detector Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

# Khởi tạo mô hình Transformer thực thụ từ Hugging Face
# Mô hình 'roberta-base-openai-detector' được fine-tune đặc biệt để phát hiện text do AI sinh ra.
print("Đang tải mô hình Mạng Nơ-ron (RoBERTa)... Quá trình này có thể mất chút thời gian trong lần đầu tiên.")
ai_detector = pipeline("text-classification", model="roberta-base-openai-detector")
print("Mô hình đã sẵn sàng không gian nhận thức.")

@app.post("/api/v1/detect")
async def detect_ai_content(payload: TextPayload):
    raw_text = payload.text
    
    # Chia văn bản thành các câu để mô hình đánh giá vi mô
    import re
    sentences = re.split(r'(?<=[.!?]) +', raw_text)
    
    results = []
    total_ai_score = 0
    valid_sentences = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
            
        # Đưa câu vào Mạng Nơ-ron để đánh giá
        prediction = ai_detector(sentence)[0]
        
        # prediction trả về dạng: {'label': 'Fake', 'score': 0.98} hoặc {'label': 'Real', 'score': 0.85}
        is_ai = prediction['label'] == 'Fake'
        confidence = prediction['score']
        
        ai_probability = confidence if is_ai else (1 - confidence)
        total_ai_score += ai_probability
        valid_sentences += 1
        
        results.append({
            "sentence": sentence,
            "is_ai": is_ai,
            "ai_probability": round(ai_probability * 100, 2)
        })
        
    global_ai_score = round((total_ai_score / valid_sentences) * 100) if valid_sentences > 0 else 0
    
    return {
        "global_ai_percentage": global_ai_score,
        "sentence_analysis": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
