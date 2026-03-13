# main.py
# Yêu cầu cài đặt môi trường: pip install fastapi uvicorn pydantic fastapi-cors

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import random
import asyncio

# Khởi tạo Không gian API
app = FastAPI(title="Linguistic Ontology AI Engine")

# Cấp quyền cho Frontend (Giao diện Web) được phép giao tiếp với Backend này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế, bạn sẽ giới hạn lại domain của mình
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa cấu trúc Bản thể Dữ liệu (Data Model)
class TextPayload(BaseModel):
    text: str

# Từ điển Tiên nghiệm (A Priori Dictionary) - Mô phỏng tri thức của Mạng nơ-ron
KNOWLEDGE_BASE = {
    'important': ['pivotal', 'crucial', 'ontologically significant', 'paramount'],
    'moreover': ['furthermore', 'additionally', 'transcending this limit'],
    'therefore': ['thus', 'consequently', 'ergo', 'henceforth'],
    'however': ['nevertheless', 'conversely', 'yet in contrast'],
    'show': ['illustrate', 'manifest', 'elucidate', 'demonstrate'],
    'use': ['utilize', 'employ', 'leverage', 'harness'],
    'clear': ['evident', 'lucid', 'transparent', 'unambiguous'],
    'very': ['profoundly', 'substantially', 'exceedingly', 'immensely']
}

@app.post("/api/v1/analyze")
async def analyze_text(payload: TextPayload):
    """
    Endpoint tiếp nhận văn bản, mô phỏng quá trình xử lý qua Mạng nơ-ron (Inference)
    và trả về danh sách các token đã được bóc tách cùng trọng số bản thể luận.
    """
    raw_text = payload.text
    
    # Mô phỏng độ trễ của việc tính toán Tensor trong Deep Learning (800ms)
    await asyncio.sleep(0.8)

    # Phân rã cấu trúc (Bảo toàn khoảng trắng)
    tokens = re.findall(r'(\S+|\s+)', raw_text)
    
    analysis_result = []
    
    for token in tokens:
        if token.isspace():
            analysis_result.append({"token": token, "type": "space", "synonyms": []})
            continue
            
        clean_word = re.sub(r'[.,!?;()]', '', token).lower()
        ontology_class = ""
        synonyms = []
        
        # Hệ thống nội suy (Heuristics)
        ai_heavy_words = list(KNOWLEDGE_BASE.keys())
        
        if clean_word in ai_heavy_words:
            ontology_class = "ai-high"
            synonyms = KNOWLEDGE_BASE[clean_word]
        elif len(clean_word) > 7:
            # Ngẫu nhiên đánh dấu một số từ dài là có rủi ro máy móc
            if random.random() > 0.6:
                ontology_class = "ai-high"
                synonyms = ['reconfigure', 'reconceptualize', 'refine']
                
        analysis_result.append({
            "token": token,
            "cleanWord": clean_word,
            "ontologyClass": ontology_class,
            "synonyms": synonyms
        })
        
    return {"status": "success", "data": analysis_result}

# Điểm khởi chạy hệ thống nội bộ
if __name__ == "__main__":
    import uvicorn
    # Vận hành máy chủ trên cổng 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
