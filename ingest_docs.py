import os
import json
import pdfplumber
import psycopg
from pgvector.psycopg import register_vector
import asyncio
from main import get_embedding

DATABASE_URL = "postgresql://risk:risk@localhost:5432/risk_db"
DOCS_FOLDER = "documents"


def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


async def ingest_all():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(DOCS_FOLDER, filename)
        print(f"Processing {filename}...")

        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                words = page.extract_words()

                for chunk_idx, chunk in enumerate(chunk_text(text)):
                    try:
                        embedding = await get_embedding(chunk)
                    except Exception as e:
                        print(f"  Skipping page {i+1} chunk {chunk_idx} — {e}")
                        continue
                    cur.execute(
                        "INSERT INTO doc_pages (doc_name, page_number, text, words, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                        (filename, i + 1, chunk, json.dumps(words), embedding),
                    )
                print(f"  Page {i+1} done")
        conn.commit()
        print(f"Done: {filename}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(ingest_all())