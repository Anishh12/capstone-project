"""Build the ChromaDB collection from docs/*.txt (fully offline)."""
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
DB_DIR = os.path.join(HERE, "chroma_db")
COLLECTION = "zepto_policies"


def chunk(text: str, size: int = 300, overlap: int = 50):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def build():
    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=ef)

    ids, docs, metas = [], [], []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        for j, ch in enumerate(chunk(text)):
            ids.append(f"{doc_id}::chunk{j}")
            docs.append(ch)
            metas.append({"doc_id": doc_id, "chunk_index": j})

    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"indexed {len(ids)} chunks across {len({m['doc_id'] for m in metas})} docs")


if __name__ == "__main__":
    build()