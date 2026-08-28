import os
import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Add this import
from pydantic import BaseModel
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
import pdfplumber
import json
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://risk:risk@localhost:5432/risk_db")

app = FastAPI(title="Customer Risk Analysis API")

# --- ADD THIS ENTIRE BLOCK ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory="documents"), name="files")


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

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 1. Search for relevant records based on the user's message
    search_req = SearchRequest(query=req.message, limit=5)
    search_results = await search(search_req)
    
    # 2. Format the retrieved context
    context = "\n".join(f"- Customer {c.customer_id}: {c.text}" for c in search_results)
    
    # 3. Build the RAG prompt
    # 3. Build the RAG prompt with STRICT instructions
    prompt = f"""You are an expert Risk Analysis AI for our company. 

### YOUR INSTRUCTIONS & RULES:
1. You must keep your answers under 3 sentences.
2. If a customer has a history of 'chargebacks' or 'fraud', you must explicitly label them as "CRITICAL RISK".
3. Always maintain a professional, cold, and analytical tone. 
4. Never guess or make up information. If the answer isn't in the provided records, say "I cannot determine that from the current records.
5. If the user asks a very vague question/ unrelated question, just return - I can't help you with that "

### DATABASE CONTEXT:
Here are the most relevant customer records from our database:
{context}

### OPERATOR QUESTION: 
{req.message}

Now, answer the operator's question based strictly on the rules and context above:"""
    
    # 4. Generate the response via Ollama
    result = await generate(GenerateRequest(prompt=prompt))
    
    return ChatResponse(reply=result.response)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 1. Search for relevant records based on the user's message
    search_req = SearchRequest(query=req.message, limit=5)
    search_results = await search(search_req)
    
    # 2. Format the retrieved context
    context = "\n".join(f"- Customer {c.customer_id}: {c.text}" for c in search_results)
    
    # 3. Build the RAG prompt
    prompt = f"""You are an AI assistant helping a customer risk operator. 
The operator asked: {req.message}

Here are the most relevant customer records from our database:
{context}

Answer the operator's question using ONLY the provided records. Be concise, helpful, and mention the customer IDs where relevant."""
    
    # 4. Generate the response via Ollama
    result = await generate(GenerateRequest(prompt=prompt))
    
    return ChatResponse(reply=result.response)

class PageQueryRequest(BaseModel):
    query: str

class PageQueryRequest(BaseModel):
    query: str
    history: list[dict] | None = None

class PageMatch(BaseModel):
    doc_name: str
    page_number: int
    answer: str
    matched_word: str | None
    box: dict | None
    page_width: float | None = None
    page_height: float | None = None


@app.post("/ask_doc", response_model=PageMatch)
async def ask_doc(req: PageQueryRequest):
    embedding = await get_embedding(req.query)
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_name, page_number, text, words FROM doc_pages ORDER BY embedding <-> %s::vector LIMIT 3",
                (embedding,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No matching page found")

    combined_context = "\n\n".join(f"[Page {r[1]}]\n{r[2]}" for r in rows)
    history_text = ""
    if req.history:
        history_text = "\n".join(f"{h['role']}: {h['text']}" for h in req.history[-4:])

    prompt = f"""You are answering questions about a document using ONLY the page excerpts below.
{f"Previous conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Page excerpts:
{combined_context}

Question: {req.query}

First give a clear, direct answer in 1-3 sentences. Then on a new line write "QUOTE:" followed by the exact phrase from the text (verbatim) that supports your answer, then "PAGE:" followed by which page number that quote came from."""

    result = await generate(GenerateRequest(prompt=prompt))
    full_response = result.response

    answer_text = full_response
    quote = ""
    matched_page, matched_words, matched_doc = rows[0][1], rows[0][3], rows[0][0]

    if "QUOTE:" in full_response:
        before, after = full_response.split("QUOTE:", 1)
        answer_text = before.strip()
        if "PAGE:" in after:
            quote, page_part = after.split("PAGE:", 1)
            quote = quote.strip()
            digits = "".join(c for c in page_part.strip()[:6] if c.isdigit())
            if digits:
                guess = int(digits)
                for r in rows:
                    if r[1] == guess:
                        matched_doc, matched_page, matched_words = r[0], r[1], r[3]
                        break
        else:
            quote = after.strip()

    matched_word, box = None, None
    if quote:
        first_word = quote.split()[0].strip(".,\"'") if quote.split() else ""
        for w in matched_words:
            if w["text"].strip(".,\"'").lower() == first_word.lower():
                matched_word = w["text"]
                box = {"x0": w["x0"], "y0": w["top"], "x1": w["x1"], "y1": w["bottom"]}
                break

    page_width, page_height = None, None
    try:
        with pdfplumber.open(f"documents/{matched_doc}") as pdf:
            p = pdf.pages[matched_page - 1]
            page_width, page_height = p.width, p.height
    except Exception:
        pass

    return PageMatch(
        doc_name=matched_doc, page_number=matched_page, answer=answer_text,
        matched_word=matched_word, box=box,
        page_width=page_width, page_height=page_height
    )