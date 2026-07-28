from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import DocumentBlock, EntityCandidate, RetrievalHit, RetrievalPacket, RetrievalQuery


QUERY_EXPANSIONS = {
    "composition": "ionizable lipid phospholipid cholesterol PEG lipid molar ratio ethanol acetate mixing formulation",
    "manufacturing": "preparation microfluidic mixing aqueous organic phase formulation",
    "hepatocyte expression": "hepatocyte eGFP GFP reporter transfection translation immunohistochemistry staining",
    "macrophage delivery": "macrophage BMDM CD163 F4/80 GFP FAPCAR transfection expression",
    "hsc therapeutic": "hepatic stellate cell activated HSC FAP FAPCAR macrophage elimination",
    "recipient cell specificity": "recipient cell CD163 F4/80 ALB Desmin SOX9 reporter expression",
}


def expand_question(question: str) -> str:
    """Add auditable biomedical synonyms without using paper-specific gold answers."""
    additions = [
        expansion for trigger, expansion in QUERY_EXPANSIONS.items()
        if trigger in question.lower()
    ]
    return question if not additions else question + " " + " ".join(additions)


class VectorBackend(Protocol):
    def build(self, blocks: list[DocumentBlock]) -> None: ...
    def search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]: ...


class TfidfVectorBackend:
    """No-download vector baseline; replaceable by Chroma/FAISS."""

    def __init__(self):
        self.blocks: list[DocumentBlock] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = None

    def build(self, blocks: list[DocumentBlock]) -> None:
        self.blocks = blocks
        self.matrix = self.vectorizer.fit_transform([block.text for block in blocks])

    def search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        candidates = [
            (self.blocks[index].block_id, float(score))
            for index, score in enumerate(scores)
            if self.blocks[index].paper_id == paper_id and score > 0
        ]
        return sorted(candidates, key=lambda row: row[1], reverse=True)[:k]


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            # The retrieval stack is a separate environment on purpose: torch
            # and faiss are large and the extraction path does not need them.
            # A bare ModuleNotFoundError here reads as a broken repository
            # rather than as "you are in the wrong venv", so say which one.
            raise ImportError(
                "SentenceTransformerBackend needs the retrieval dependencies, "
                "which live in a separate environment:\n"
                "    python3 -m venv .venv-rag\n"
                "    .venv-rag/bin/pip install -r requirements-rag.txt\n"
                "Then run this module with .venv-rag/bin/python. The extraction "
                "path does not need them and runs from .venv."
            ) from error
        self.model = SentenceTransformer(model_name)
        self.blocks: list[DocumentBlock] = []
        self.matrix: np.ndarray | None = None

    def build(self, blocks: list[DocumentBlock]) -> None:
        self.blocks = blocks
        self.matrix = self.model.encode(
            [block.text for block in blocks], normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ query_vector
        candidates = [
            (self.blocks[index].block_id, float(score))
            for index, score in enumerate(scores)
            if self.blocks[index].paper_id == paper_id
        ]
        return sorted(candidates, key=lambda row: row[1], reverse=True)[:k]


class FaissSentenceTransformerBackend(SentenceTransformerBackend):
    """FAISS dense retrieval with a paper filter and exact NumPy fallback."""

    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        super().__init__(model_name)
        self.faiss_index = None

    def build(self, blocks: list[DocumentBlock]) -> None:
        super().build(blocks)
        try:
            import faiss
        except ImportError:
            return
        self.faiss_index = faiss.IndexFlatIP(self.matrix.shape[1])
        self.faiss_index.add(self.matrix.astype("float32"))

    def search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]:
        if self.faiss_index is None:
            return super().search(query, paper_id, k)
        query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        # Search the full small corpus, then enforce the hard paper boundary.
        scores, indices = self.faiss_index.search(query_vector, len(self.blocks))
        candidates = [
            (self.blocks[index].block_id, float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0 and self.blocks[index].paper_id == paper_id
        ]
        return candidates[:k]


class ChromaSentenceTransformerBackend(SentenceTransformerBackend):
    """Chroma adapter; deliberately lazy so core retrieval has no server dependency."""

    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        super().__init__(model_name)
        self.collection = None

    def build(self, blocks: list[DocumentBlock]) -> None:
        super().build(blocks)
        try:
            import chromadb
        except ImportError:
            return
        client = chromadb.EphemeralClient()
        self.collection = client.get_or_create_collection("ai_lnp")
        self.collection.add(
            ids=[block.block_id for block in blocks],
            embeddings=self.matrix.tolist(),
            metadatas=[{"paper_id": block.paper_id} for block in blocks],
        )

    def search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]:
        if self.collection is None:
            return super().search(query, paper_id, k)
        vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"paper_id": paper_id},
        )
        return [
            (block_id, 1.0 - float(distance))
            for block_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]


class HybridIndex:
    def __init__(self, db_path: Path, vector_backend: VectorBackend):
        self.db_path = db_path
        self.vector_backend = vector_backend
        self.blocks: dict[str, DocumentBlock] = {}
        self.entities: dict[str, list[EntityCandidate]] = defaultdict(list)
        self.block_order: dict[str, int] = {}
        self.ordered_blocks: list[DocumentBlock] = []

    def build(self, blocks: list[DocumentBlock], entities: Iterable[EntityCandidate] = ()) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.executescript("""
            DROP TABLE IF EXISTS blocks;
            DROP TABLE IF EXISTS blocks_fts;
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE blocks_fts USING fts5(
                block_id UNINDEXED, paper_id UNINDEXED, text, section_path,
                tokenize='porter unicode61'
            );
        """)
        for block in blocks:
            payload = block.model_dump_json()
            connection.execute("INSERT INTO blocks VALUES (?,?,?)", (block.block_id, block.paper_id, payload))
            connection.execute(
                "INSERT INTO blocks_fts(block_id,paper_id,text,section_path) VALUES (?,?,?,?)",
                (block.block_id, block.paper_id, block.text, block.section_path),
            )
            self.blocks[block.block_id] = block
        self.ordered_blocks = blocks
        self.block_order = {block.block_id: index for index, block in enumerate(blocks)}
        connection.commit()
        connection.close()
        for entity in entities:
            self.entities[entity.block_id].append(entity)
        self.vector_backend.build(blocks)

    def lexical_search(self, query: str, paper_id: str, k: int) -> list[tuple[str, float]]:
        terms = [token for token in query.replace('"', " ").split() if len(token) > 2]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        connection = sqlite3.connect(self.db_path)
        rows = connection.execute(
            "SELECT block_id, -bm25(blocks_fts) score FROM blocks_fts WHERE blocks_fts MATCH ? AND paper_id=? ORDER BY bm25(blocks_fts) LIMIT ?",
            (fts_query, paper_id, k),
        ).fetchall()
        connection.close()
        return [(row[0], float(row[1])) for row in rows]

    def retrieve(self, query: RetrievalQuery, k: int = 6, rrf_k: int = 60) -> RetrievalPacket:
        expanded_question = expand_question(query.question)
        lexical = self.lexical_search(expanded_question, query.paper_id, k * 2)
        vector = self.vector_backend.search(expanded_question, query.paper_id, k * 2)
        fused: dict[str, float] = defaultdict(float)
        lexical_scores, vector_scores = dict(lexical), dict(vector)
        for rank, (block_id, _) in enumerate(lexical, 1):
            fused[block_id] += 1 / (rrf_k + rank)
        for rank, (block_id, _) in enumerate(vector, 1):
            fused[block_id] += 1 / (rrf_k + rank)

        # Experimental details and results frequently span neighboring paragraphs.
        # Give immediate same-source neighbors of the strongest hits a bounded bonus;
        # they still compete on score and can never cross a paper or document boundary.
        for block_id in sorted(fused, key=fused.get, reverse=True)[:4]:
            block = self.blocks[block_id]
            position = self.block_order[block_id]
            for neighbor_position in (position - 1, position + 1):
                if not 0 <= neighbor_position < len(self.ordered_blocks):
                    continue
                neighbor = self.ordered_blocks[neighbor_position]
                if neighbor.paper_id == block.paper_id and neighbor.source_path == block.source_path:
                    fused[neighbor.block_id] += fused[block_id] * 0.18
        ranked = sorted(fused, key=fused.get, reverse=True)
        if query.required_entity_types:
            ranked.sort(
                key=lambda block_id: (
                    not bool(set(query.required_entity_types) & {x.entity_type for x in self.entities.get(block_id, [])}),
                    -fused[block_id],
                )
            )
        selected = ranked[:k]
        # If a paper has both main XML and supplements, avoid allowing one source
        # kind to monopolize every result. This is critical for composition tables.
        paper_kinds = {block.source_kind for block in self.blocks.values() if block.paper_id == query.paper_id}
        if query.field_group in {"formulation", "component"} and len(paper_kinds) > 1:
            selected_kinds = {self.blocks[block_id].source_kind for block_id in selected}
            for missing_kind in paper_kinds - selected_kinds:
                replacement = next(
                    (block_id for block_id in ranked[k:] if self.blocks[block_id].source_kind == missing_kind),
                    None,
                )
                if replacement:
                    selected[-1] = replacement
        hits = []
        for block_id in selected:
            block = self.blocks[block_id]
            hits.append(RetrievalHit(
                query_id=query.query_id, block_id=block_id, paper_id=block.paper_id,
                text=block.text, section_path=block.section_path, source_path=block.source_path,
                source_kind=block.source_kind, block_type=block.block_type,
                page_number=block.page_number, xml_element_id=block.xml_element_id,
                table_number=block.table_number, figure_number=block.figure_number,
                lexical_score=lexical_scores.get(block_id),
                vector_score=vector_scores.get(block_id), fused_score=fused[block_id],
                entity_types=sorted({x.entity_type for x in self.entities.get(block_id, [])}),
            ))
        warnings = []
        if not hits:
            warnings.append("No retrieval hits; downstream extraction must abstain.")
        return RetrievalPacket(
            query=query, hits=hits, retrieval_methods=["sqlite_fts5_bm25", type(self.vector_backend).__name__],
            warnings=warnings,
        )

    def expand_hierarchy_context(
        self,
        packet: RetrievalPacket,
        *,
        seed_count: int = 3,
        neighbors_per_seed: int = 1,
        max_hits: int = 10,
    ) -> RetrievalPacket:
        """Add bounded neighboring blocks without crossing a source or section."""
        selected_ids = {hit.block_id for hit in packet.hits}
        expanded = list(packet.hits)
        for seed in packet.hits[:seed_count]:
            block = self.blocks[seed.block_id]
            position = self.block_order[seed.block_id]
            candidates: list[DocumentBlock] = []
            for distance in range(1, neighbors_per_seed + 1):
                for neighbor_position in (position - distance, position + distance):
                    if not 0 <= neighbor_position < len(self.ordered_blocks):
                        continue
                    neighbor = self.ordered_blocks[neighbor_position]
                    if neighbor.paper_id != block.paper_id or neighbor.source_path != block.source_path:
                        continue
                    # PDF-derived blocks (legacy page dumps and uniparse
                    # elements alike) are bounded by page adjacency rather than
                    # by section, because a table and the sentence that cites it
                    # routinely sit under different headings.
                    if block.source_kind in {"pdf", "uniparse"}:
                        if (
                            block.page_number is None
                            or neighbor.page_number is None
                            or abs(block.page_number - neighbor.page_number) > 1
                        ):
                            continue
                    elif neighbor.section_path != block.section_path:
                        continue
                    candidates.append(neighbor)
            for neighbor in candidates:
                if neighbor.block_id in selected_ids or len(expanded) >= max_hits:
                    continue
                selected_ids.add(neighbor.block_id)
                expanded.append(RetrievalHit(
                    query_id=packet.query.query_id,
                    block_id=neighbor.block_id,
                    paper_id=neighbor.paper_id,
                    text=neighbor.text,
                    section_path=neighbor.section_path,
                    source_path=neighbor.source_path,
                    source_kind=neighbor.source_kind,
                    block_type=neighbor.block_type,
                    page_number=neighbor.page_number,
                    xml_element_id=neighbor.xml_element_id,
                    table_number=neighbor.table_number,
                    figure_number=neighbor.figure_number,
                    fused_score=seed.fused_score * 0.18,
                    entity_types=sorted({
                        value.entity_type
                        for value in self.entities.get(neighbor.block_id, [])
                    }),
                ))
        return packet.model_copy(update={
            "hits": expanded,
            "retrieval_methods": packet.retrieval_methods + [
                "hierarchy_bounded_neighbor_expansion"
            ],
        })


def load_blocks(corpus_root: Path) -> list[DocumentBlock]:
    blocks = []
    for path in sorted(corpus_root.glob("GP-*.blocks.jsonl")):
        blocks.extend(DocumentBlock.model_validate_json(line) for line in path.read_text().splitlines() if line)
    return blocks
