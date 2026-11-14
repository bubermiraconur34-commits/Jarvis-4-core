
import os
import secrets
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="JARVIS 4.0")

API_KEY = os.getenv("OPENAI_API_KEY")
ACCESS_TOKEN = os.getenv("JARVIS_ACCESS_TOKEN")
MODEL = os.getenv("JARVIS_MODEL", "gpt-5.5")

if not API_KEY or not ACCESS_TOKEN:
    raise RuntimeError("Sunucu güvenlik ayarları eksik!")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
Sen J.A.R.V.I.S. 4.0'sın.
Iron Man filmindeki JARVIS gibi davran.
Kullanıcıya Türkçe cevap ver.
Kibar, zeki, sakin ve yardımsever ol.
Kullanıcıya Komutanım diye hitap et.
Soruları dikkatlice anla.
Bilmediğin bilgileri uydurma.
Gerçekleştirmediğin işlemleri yaptığını söyleme.
İzinsiz cihaz işlemi yapma.
"""

class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=20)

class ChatResponse(BaseModel):
    reply: str
    status: str

def verify_access(authorization):
    if not authorization:
        raise HTTPException(401, "Giriş anahtarı gerekli")

    parts = authorization.split(" ", 1)

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Geçersiz erişim")

    if not secrets.compare_digest(parts[1], ACCESS_TOKEN):
        raise HTTPException(403, "Erişim reddedildi")

@app.get("/")
def home():
    return {
        "system": "JARVIS 4.0",
        "status": "ONLINE"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(
    data: ChatRequest,
    authorization: str | None = Header(default=None)
):
    verify_access(authorization)

    messages = []

    for item in data.history[-20:]:
        if item.role not in ("user", "assistant"):
            raise HTTPException(400, "Geçersiz mesaj rolü")

        messages.append({
            "role": item.role,
            "content": item.content
        })

    messages.append({
        "role": "user",
        "content": data.message
    })

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=messages
        )

        answer = response.output_text

        return ChatResponse(
            reply=answer or "Komutanım, cevap oluşturulamadı.",
            status="SUCCESS"
        )

    except Exception:
        logging.exception("Yapay zeka hatası")
        raise HTTPException(
            502,
            "JARVIS yapay zeka bağlantısında hata oluştu."
        )
