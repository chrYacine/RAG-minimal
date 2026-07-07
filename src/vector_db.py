from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CORPUS_CSV_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)


@dataclass(frozen=True)
class CorpusChunk:
    """Single knowledge-base unit stored in ChromaDB."""

    id: str
    text: str
    source: str
    categorie: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CorpusChunk":
        required_columns = {"id", "text", "source", "categorie"}
        missing_columns = required_columns.difference(row.keys())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Chunk is missing required columns: {missing}")

        chunk_id = str(row["id"]).strip()
        text = str(row["text"]).strip()
        source = str(row["source"]).strip()
        categorie = str(row["categorie"]).strip()

        if not chunk_id or not text:
            raise ValueError("Chunk id and text must be non-empty.")

        return cls(id=chunk_id, text=text, source=source, categorie=categorie)

    def metadata(self) -> dict[str, str]:
        return {"source": self.source, "categorie": self.categorie}


@dataclass(frozen=True)
class RetrievedChunk:
    """Result returned by a vector search."""

    id: str
    text: str
    source: str
    categorie: str
    distance: float | None

    @property
    def similarity(self) -> float | None:
        if self.distance is None:
            return None
        return 1 - self.distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "categorie": self.categorie,
            "metadata": {"source": self.source, "categorie": self.categorie},
            "distance": self.distance,
            "similarity": self.similarity,
        }


class BaseTextEncoder(ABC):
    """Abstract text encoder so tests can inject a fake encoder later."""

    model_name: str

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEncoder(BaseTextEncoder):
    """Sentence-transformers implementation used by the TP."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
        )
        return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)


class BaseVectorDB(ABC):
    """Contract consumed by the RAG pipeline implemented by Adrien."""

    @abstractmethod
    def retrieve(self, question: str, n: int = 3) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class VectorDB(BaseVectorDB):
    """Persistent ChromaDB vector store for the mini-RAG corpus.

    Behavior required by the TP:
    - if the collection already exists, reload it without re-indexing;
    - if it does not exist and chunks are provided, create and persist it;
    - if it does not exist and no chunks are provided, fail explicitly.
    """

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        chunks: Sequence[CorpusChunk | Mapping[str, Any]] | None = None,
        encoder: BaseTextEncoder | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        recreate: bool = False,
    ) -> None:
        from chromadb import PersistentClient

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=str(self.persist_directory))

        if recreate and self._collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        if self._collection_exists(collection_name):
            self.collection = self.client.get_collection(collection_name)
            stored_model_name = self._stored_embedding_model_name(default=embedding_model_name)
            self.encoder = encoder or SentenceTransformerEncoder(model_name=stored_model_name)
            return

        if chunks is None:
            raise ValueError(
                "No persisted ChromaDB collection found. Provide chunks to create it first."
            )

        self.encoder = encoder or SentenceTransformerEncoder(model_name=embedding_model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_model_name": self.encoder.model_name,
                "hnsw:space": "cosine",
            },
        )
        self.add_chunks(chunks)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path = CORPUS_CSV_PATH,
        **kwargs: Any,
    ) -> "VectorDB":
        return cls(chunks=load_chunks_from_csv(csv_path), **kwargs)

    def _collection_exists(self, collection_name: str) -> bool:
        existing_collections = self.client.list_collections()
        existing_names = {
            item.name if hasattr(item, "name") else str(item)
            for item in existing_collections
        }
        return collection_name in existing_names

    def _stored_embedding_model_name(self, default: str) -> str:
        metadata = self.collection.metadata or {}
        return str(metadata.get("embedding_model_name", default))

    def add_chunks(self, chunks: Sequence[CorpusChunk | Mapping[str, Any]]) -> None:
        normalized_chunks = [normalize_chunk(chunk) for chunk in chunks]
        if not normalized_chunks:
            raise ValueError("Cannot index an empty corpus.")

        ids = [chunk.id for chunk in normalized_chunks]
        documents = [chunk.text for chunk in normalized_chunks]
        metadatas = [chunk.metadata() for chunk in normalized_chunks]
        embeddings = self.encoder.encode(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def retrieve_chunks(self, question: str, n: int = 3) -> list[RetrievedChunk]:
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")
        if n <= 0:
            raise ValueError("n must be greater than 0.")

        query_embedding = self.encoder.encode([question.strip()])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            retrieved_chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(document),
                    source=str(metadata.get("source", "unknown")),
                    categorie=str(metadata.get("categorie", "unknown")),
                    distance=float(distance) if distance is not None else None,
                )
            )
        return retrieved_chunks

    def retrieve(self, question: str, n: int = 3) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.retrieve_chunks(question=question, n=n)]

    def count(self) -> int:
        return int(self.collection.count())


def normalize_chunk(chunk: CorpusChunk | Mapping[str, Any]) -> CorpusChunk:
    if isinstance(chunk, CorpusChunk):
        return chunk
    return CorpusChunk.from_mapping(chunk)


def load_chunks_from_csv(csv_path: str | Path = CORPUS_CSV_PATH) -> list[CorpusChunk]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Corpus CSV is empty or missing headers.")
        return [CorpusChunk.from_mapping(row) for row in reader]