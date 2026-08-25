import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

app = FastAPI(title="Customer Risk Analysis API")


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    response: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    payload = {
        "model": "llama3.1:8b",
        "prompt": req.prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Ollama is unreachable. Is it running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=503, detail=f"Ollama returned an error: {e}")

    data = r.json()
    return GenerateResponse(response=data.get("response", ""))