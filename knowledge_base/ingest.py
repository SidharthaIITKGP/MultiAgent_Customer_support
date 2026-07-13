"""
knowledge_base/ingest.py

Chunks all markdown docs in knowledge_base/docs/ and embeds them into a
persistent ChromaDB collection using a local sentence-transformer model.

Run once before starting the system:
    python knowledge_base/ingest.py

Or re-run after adding new docs:
    python knowledge_base/ingest.py --reset
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "knowledge_base"

CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 50      # characters


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks, preserving source metadata."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        # Try to break at a paragraph/sentence boundary rather than mid-word
        if end < len(text):
            # Prefer paragraph break
            para_break = chunk_text.rfind("\n\n")
            if para_break > CHUNK_SIZE // 2:
                end = start + para_break + 2
                chunk_text = text[start:end]
            else:
                # Fall back to sentence break
                sent_break = max(chunk_text.rfind(". "), chunk_text.rfind(".\n"))
                if sent_break > CHUNK_SIZE // 2:
                    end = start + sent_break + 2
                    chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text.strip(),
            "source": source,
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        start = end - CHUNK_OVERLAP  # overlap for context continuity

    return [c for c in chunks if len(c["text"]) > 50]  # drop tiny trailing chunks


def ingest(reset: bool = False) -> None:
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Use a local sentence-transformer model — no API calls, no cost
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    if reset:
        print(f"Resetting collection '{COLLECTION_NAME}'...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection already has {existing_count} chunks. Use --reset to reingest.")
        return

    doc_files = sorted(DOCS_DIR.glob("*.md"))
    if not doc_files:
        print(f"ERROR: No .md files found in {DOCS_DIR}")
        sys.exit(1)

    print(f"Found {len(doc_files)} documents to ingest...")

    all_ids, all_texts, all_metadatas = [], [], []

    for doc_path in doc_files:
        text = doc_path.read_text(encoding="utf-8")
        source = doc_path.name
        chunks = chunk_text(text, source)

        for chunk in chunks:
            chunk_id = f"{source}::chunk_{chunk['chunk_index']}"
            all_ids.append(chunk_id)
            all_texts.append(chunk["text"])
            all_metadatas.append({
                "source": source,
                "chunk_index": chunk["chunk_index"],
            })

        print(f"  {source}: {len(chunks)} chunks")

    # Upsert in batches of 100 to stay within ChromaDB limits
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        collection.upsert(
            ids=all_ids[i : i + batch_size],
            documents=all_texts[i : i + batch_size],
            metadatas=all_metadatas[i : i + batch_size],
        )

    final_count = collection.count()
    print(f"\n✓ Ingestion complete. Collection '{COLLECTION_NAME}' now has {final_count} chunks.")
    print(f"  ChromaDB path: {Path(CHROMA_DB_PATH).resolve()}")


def query_test(query: str, n_results: int = 3) -> None:
    """Quick sanity check — retrieve top-k chunks for a query."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "distances", "metadatas"])

    print(f"\nQuery: '{query}'")
    print(f"Top {n_results} results:")
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        similarity = 1 - dist  # cosine distance → similarity
        print(f"\n  [{i+1}] Source: {meta['source']} (chunk {meta['chunk_index']}) | similarity: {similarity:.3f}")
        print(f"  {doc[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base docs into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Delete and re-create the collection")
    parser.add_argument("--test-query", type=str, default=None, help="Run a test query after ingestion")
    args = parser.parse_args()

    ingest(reset=args.reset)

    if args.test_query:
        query_test(args.test_query)
    else:
        # Always run a default sanity check
        query_test("how do I get a refund?")
        query_test("my account is locked")
        query_test("error E-4023 login crash")
