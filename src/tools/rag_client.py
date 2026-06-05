"""
RAG (Retrieval-Augmented Generation) client using ChromaDB.
Provides: document ingestion, chunking, embedding, and retrieval.
"""
from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from src.config import get_settings

_client: chromadb.Client | None = None
_collection: chromadb.Collection | None = None
_COLLECTION_NAME = "drug_repurposing_literature"


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection with sentence-transformers embeddings."""
    global _client, _collection
    if _collection is not None:
        return _collection

    settings = get_settings()
    persist_dir = str(settings.chroma_db_path)

    # Persistent ChromaDB client
    _client = chromadb.PersistentClient(path=persist_dir)

    # Use sentence-transformers for embedding (local, no API key needed)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB collection '{_COLLECTION_NAME}' ready "
        f"({_collection.count()} documents)"
    )
    return _collection


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _make_doc_id(text: str, prefix: str = "") -> str:
    """Generate a stable document ID from content hash."""
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}_{h}" if prefix else h


def ingest_articles(articles: list[dict[str, Any]]) -> int:
    """
    Ingest PubMed/Europe PMC articles into ChromaDB.

    Args:
        articles: List of article dicts with keys: pmid, title, abstract, year, url.

    Returns:
        Number of new documents added.
    """
    collection = _get_collection()
    settings = get_settings()

    documents = []
    metadatas = []
    ids = []

    for article in articles:
        text = f"{article.get('title', '')}. {article.get('abstract', '')}"
        if not text.strip() or len(text) < 50:
            continue

        # Chunk the text
        chunks = _chunk_text(
            text,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )

        for i, chunk in enumerate(chunks):
            doc_id = _make_doc_id(chunk, prefix=f"pmid{article.get('pmid', 'x')}")
            documents.append(chunk)
            ids.append(f"{doc_id}_{i}")
            metadatas.append({
                "pmid": str(article.get("pmid", "")),
                "title": article.get("title", "")[:200],
                "year": str(article.get("year", "")),
                "url": article.get("url", ""),
                "journal": article.get("journal", ""),
                "source": "pubmed",
            })

    if not documents:
        return 0

    # ChromaDB handles deduplication by ID (upsert)
    try:
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"RAG: ingested {len(documents)} chunks from {len(articles)} articles")
        return len(documents)
    except Exception as e:
        logger.warning(f"RAG ingestion error: {e}")
        return 0


def query_literature(
    query: str,
    top_k: int | None = None,
    filter_metadata: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant literature passages for a query.

    Args:
        query: Natural language query (e.g. "Metformin mechanism in Alzheimer's")
        top_k: Number of results to return.
        filter_metadata: Optional ChromaDB metadata filter.

    Returns:
        List of dicts with: text, score, pmid, title, year, url.
    """
    collection = _get_collection()
    settings = get_settings()
    k = top_k or settings.rag_top_k

    if collection.count() == 0:
        logger.warning("RAG: collection is empty. Ingest articles first.")
        return []

    try:
        where = filter_metadata if filter_metadata else None
        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        passages = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - distance)  # Convert cosine distance to similarity
            meta = results["metadatas"][0][i]
            passages.append({
                "text": doc,
                "relevance_score": round(score, 4),
                "pmid": meta.get("pmid", ""),
                "title": meta.get("title", ""),
                "year": meta.get("year", ""),
                "url": meta.get("url", ""),
                "journal": meta.get("journal", ""),
            })

        logger.info(f"RAG: retrieved {len(passages)} passages for query")
        return sorted(passages, key=lambda p: p["relevance_score"], reverse=True)

    except Exception as e:
        logger.warning(f"RAG query failed: {e}")
        return []


def get_collection_stats() -> dict[str, Any]:
    """Return statistics about the current ChromaDB collection."""
    try:
        collection = _get_collection()
        return {
            "total_documents": collection.count(),
            "collection_name": _COLLECTION_NAME,
        }
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


def clear_collection() -> None:
    """Clear all documents from the collection (useful for testing)."""
    global _collection
    client = chromadb.PersistentClient(path=str(get_settings().chroma_db_path))
    try:
        client.delete_collection(_COLLECTION_NAME)
        _collection = None
        logger.info("RAG: collection cleared")
    except Exception as e:
        logger.warning(f"RAG: could not clear collection: {e}")
