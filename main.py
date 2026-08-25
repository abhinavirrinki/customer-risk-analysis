import os
import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://risk:risk@localhost:5432/risk_db")

app = FastAPI(title="Customer Risk Analysis API")


def get_db():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]


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


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


class SearchResult(BaseModel):
    id: int
    customer_id: str
    text: str


@app.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest):
    embedding = await get_embedding(req.query)
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, customer_id, text FROM cases ORDER BY embedding <-> %s::vector LIMIT %s",
                (embedding, req.limit),
            )
            rows = cur.fetchall()
        return [SearchResult(id=r[0], customer_id=r[1], text=r[2]) for r in rows]
    finally:
        conn.close()


class ExplainResponse(BaseModel):
    customer_id: str
    risk_score: str
    explanation: str
    similar_cases: list[SearchResult]


@app.get("/explain/{customer_id}", response_model=ExplainResponse)
async def explain(customer_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT text FROM cases WHERE customer_id = %s", (customer_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")

    combined_text = " ".join(r[0] for r in rows)
    risk_score = "High" if len(rows) >= 2 else "Low"

    search_results = await search(SearchRequest(query=combined_text, limit=3))

    context = "\n".join(f"- {c.text}" for c in search_results)
    prompt = f"""A customer has been flagged with risk score: {risk_score}.
Their case history: {combined_text}

Similar past cases:
{context}

In 2-3 sentences, explain why this customer might be flagged as {risk_score} risk, referencing the similar cases."""

    result = await generate(GenerateRequest(prompt=prompt))

    return ExplainResponse(
        customer_id=customer_id,
        risk_score=risk_score,
        explanation=result.response,
        similar_cases=search_results,
    )