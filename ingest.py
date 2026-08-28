from pgvector.psycopg import register_vector
import httpx
import psycopg
from sample_data import SAMPLE_CASES
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434"
DATABASE_URL = "postgresql://risk:risk@localhost:5432/risk_db"

def get_embedding(text: str) -> list[float]:
    r = httpx.post(f"{OLLAMA_URL}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text}, timeout=60.0)
    r.raise_for_status()
    return r.json()["embedding"]

def main():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for case in SAMPLE_CASES:
                emb = get_embedding(case["text"])
                cur.execute(
                    "INSERT INTO cases (customer_id, text, embedding) VALUES (%s, %s, %s)",
                    (case["customer_id"], case["text"], emb),
                )
                print(f"inserted case for {case['customer_id']}")
        conn.commit()

if __name__ == "__main__":
    main()