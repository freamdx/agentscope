# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the AnalyticDBMySQLStore class.

AnalyticDB for MySQL is a hosted vector database, so we cannot spin
one up in CI the way the Qdrant tests use an in-process instance.
Instead we inject a **fake pymysql connection** that interprets the
same SQL the store emits and simulates the vector semantics
(``ARRAY<FLOAT>`` columns, ``cosine_similarity`` / ``l2_distance``
functions, ``JSON_EXTRACT`` payload predicates) entirely in Python.

This keeps the store's real SQL-generation + result-parsing logic
under test without a live database.
"""
import json
import math
import re
from contextlib import AsyncExitStack
from typing import Any
from unittest.async_case import IsolatedAsyncioTestCase

from utils import AnyString

from agentscope.message import TextBlock
from agentscope.rag import (
    AnalyticDBMySQLStore,
    Chunk,
    VectorRecord,
    VectorSearchResult,
)

# Matches the ``cosine_similarity(embedding, '[...]')`` / ``l2_distance(...)``
# call emitted by ``AnalyticDBMySQLStore.search``.
_DISTANCE_FUNC_RE = re.compile(
    r"(cosine_similarity|l2_distance)\(embedding,\s*'([^']+)'\)",
)


# ======================================================================
# In-memory fake of a pymysql connection / cursor
# =======================================================================


class _FakeCursor:
    """A minimal cursor that interprets the SQL emitted by
    :class:`AnalyticDBMySQLStore`."""

    def __init__(self, conn: "_FakeConnection") -> None:
        self._conn = conn
        self.description: list[tuple] | None = None
        self._rows: list[tuple] = []

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _qualified_name(sql: str) -> str:
        """Extract the table (collection) name from a ``db`.`tbl`` ref."""
        match = re.search(r"`[^`]+`\.`([^`]+)`", sql)
        if match is None:
            raise AssertionError(f"Cannot parse table name from: {sql}")
        return match.group(1)

    @staticmethod
    def _substitute(sql: str, params: Any) -> str:
        """Substitute ``%s`` / ``%(name)s`` placeholders like pymysql."""
        if params is None:
            return sql
        if isinstance(params, dict):
            return sql % params
        if isinstance(params, (list, tuple)):
            return sql % tuple(params)
        return sql % (params,)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _l2(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _parse_metadata_filter(self, sql: str) -> dict[str, Any]:
        """Extract ``{key: value}`` pairs from the JSON_EXTRACT WHERE."""
        result: dict[str, Any] = {}
        for match in re.finditer(
            r"JSON_EXTRACT\(chunk, '\$\.metadata\.([^']+)'\)\s*=\s*"
            r"CAST\((.+?) AS JSON\)",
            sql,
        ):
            key = match.group(1)
            value = json.loads(match.group(2))
            result[key] = value
        return result

    # -- execute -------------------------------------------------------
    def execute(self, sql: str, params: Any = None) -> None:
        """Dispatch a single SQL statement, populating ``description`` /
        ``_rows`` for SELECTs, or mutating the in-memory tables otherwise.
        """
        sql = self._substitute(sql, params)
        tables = self._conn._tables
        lowered = sql.strip()

        # CREATE TABLE IF NOT EXISTS `db`.`name` (... ARRAY<FLOAT>(d) ...)
        if lowered.startswith("CREATE TABLE"):
            name = self._qualified_name(sql)
            dim_match = re.search(r"ARRAY<FLOAT\((\d+)\)>", sql)
            dimensions = int(dim_match.group(1)) if dim_match else 0
            tables[name] = {"dimensions": dimensions, "rows": []}
            self.description = None
            self._rows = []
            return

        # DROP TABLE IF EXISTS `db`.`name`
        if lowered.startswith("DROP TABLE"):
            name = self._qualified_name(sql)
            tables.pop(name, None)
            self.description = None
            self._rows = []
            return

        # SHOW TABLES LIKE 'name'
        if lowered.startswith("SHOW TABLES"):
            match = re.search(r"LIKE '([^']+)'", sql)
            name = match.group(1) if match else ""
            self.description = [("Tables_in_db",)]
            self._rows = [(name,)] if name in tables else []
            return

        # REPLACE INTO `db`.`name` (cols) VALUES (%(id)s, ...)  [single row]
        if lowered.startswith("REPLACE INTO"):
            name = self._qualified_name(sql)
            rows = tables[name]["rows"]
            assert isinstance(params, dict)
            row = {
                "id": params["id"],
                "embedding": json.loads(params["embedding"]),
                "document_id": params["document_id"],
                "chunk": params["chunk"],
            }
            # REPLACE semantics: upsert by id.
            for i, existing in enumerate(rows):
                if existing["id"] == row["id"]:
                    rows[i] = row
                    break
            else:
                rows.append(row)
            self.description = None
            self._rows = []
            return

        # DELETE FROM `db`.`name` WHERE document_id = <value>
        if lowered.startswith("DELETE FROM"):
            name = self._qualified_name(sql)
            rows = tables[name]["rows"]
            doc_id = params  # already substituted into sql via %s
            # params may be a single value or tuple; extract the value.
            if isinstance(params, (list, tuple)):
                doc_id = params[0]
            tables[name]["rows"] = [
                r for r in rows if r["document_id"] != doc_id
            ]
            self.description = None
            self._rows = []
            return

        # SELECT ... — search or list_documents
        if lowered.startswith("SELECT"):
            name = self._qualified_name(sql)
            rows = tables.get(name, {}).get("rows", [])
            meta_filter = self._parse_metadata_filter(sql)

            def _matches(row: dict) -> bool:
                chunk = json.loads(row["chunk"])
                metadata = chunk.get("metadata", {})
                return all(
                    metadata.get(k) == v for k, v in meta_filter.items()
                )

            filtered = [r for r in rows if _matches(r)]

            if "cosine_similarity" in sql or "l2_distance" in sql:
                # --- search ---
                vec_match = _DISTANCE_FUNC_RE.search(sql)
                assert vec_match is not None
                func = vec_match.group(1)
                query_vec = json.loads(vec_match.group(2))
                metric = (
                    self._cosine if func == "cosine_similarity" else self._l2
                )
                scored = [
                    (metric(r["embedding"], query_vec), r) for r in filtered
                ]
                if func == "cosine_similarity":
                    scored.sort(key=lambda x: x[0], reverse=True)
                else:
                    scored.sort(key=lambda x: x[0])
                limit_match = re.search(r"LIMIT (\d+)", sql)
                limit = (
                    int(limit_match.group(1)) if limit_match else len(scored)
                )
                scored = scored[:limit]
                self.description = [
                    ("id",),
                    ("document_id",),
                    ("chunk",),
                    ("distance",),
                ]
                self._rows = [
                    (r["id"], r["document_id"], r["chunk"], score)
                    for score, r in scored
                ]
                return

            # --- list_documents ---
            self.description = [("document_id",), ("chunk",)]
            self._rows = [(r["document_id"], r["chunk"]) for r in filtered]
            return

        raise AssertionError(f"Unhandled SQL: {sql}")

    def executemany(self, sql: str, seq: list[Any]) -> None:
        """Execute the same statement once per parameter set in ``seq``."""
        for params in seq:
            self.execute(sql, params)

    def fetchall(self) -> list[tuple]:
        """Return the rows materialised by the previous SELECT."""
        return self._rows

    def close(self) -> None:
        """No-op — the cursor holds no external resources."""


class _FakeConnection:
    """A minimal in-memory stand-in for a ``pymysql`` connection."""

    def __init__(self) -> None:
        self._tables: dict[str, dict] = {}

    def cursor(self) -> _FakeCursor:
        """Return a fresh cursor bound to this connection."""
        return _FakeCursor(self)

    def commit(self) -> None:
        """No-op — writes are already in memory."""

    def rollback(self) -> None:
        """No-op — writes are already in memory."""

    def close(self) -> None:
        """No-op — there is nothing to close."""


# ======================================================================
# Test helpers
# =======================================================================


def _make_store() -> AnalyticDBMySQLStore:
    """Build an :class:`AnalyticDBMySQLStore` backed by a fake connection."""
    store = AnalyticDBMySQLStore.__new__(AnalyticDBMySQLStore)
    store._database = "rag"
    store._distance = "COSINE"
    store._conn_kwargs = {}
    store._conn = _FakeConnection()
    return store


def _make_record(
    text: str,
    vector: list[float],
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> VectorRecord:
    """Build a VectorRecord for testing."""
    return VectorRecord(
        vector=vector,
        document_id=document_id,
        chunk=Chunk(
            content=TextBlock(text=text),
            source=f"{document_id}.txt",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        ),
    )


def _dump_results(results: list[VectorSearchResult]) -> list[dict]:
    """Convert search results into plain dicts for whole-structure
    comparison."""
    return [result.model_dump() for result in results]


# ======================================================================
# Tests
# =======================================================================


class AnalyticDBMySQLStoreTest(IsolatedAsyncioTestCase):
    """The test cases for the AnalyticDBMySQLStore class."""

    async def asyncSetUp(self) -> None:
        """Create a store backed by a fake connection before each test."""
        self._exit_stack = AsyncExitStack()
        self.store = await self._exit_stack.enter_async_context(
            _make_store(),
        )

    async def asyncTearDown(self) -> None:
        """Close the store after each test."""
        await self._exit_stack.aclose()

    async def test_collection_lifecycle(self) -> None:
        """Collections (tables) can be created, checked, and deleted."""
        self.assertEqual(await self.store.has_collection("kb-1"), False)

        await self.store.create_collection("kb-1", dimensions=3)
        self.assertEqual(await self.store.has_collection("kb-1"), True)

        # Creating an existing collection is a no-op
        await self.store.create_collection("kb-1", dimensions=3)
        self.assertEqual(await self.store.has_collection("kb-1"), True)

        await self.store.delete_collection("kb-1")
        self.assertEqual(await self.store.has_collection("kb-1"), False)

    async def test_insert_and_search(self) -> None:
        """Inserted records are searchable, ordered by similarity."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "Hello world!",
                    [1.0, 0.0, 0.0],
                    document_id="doc-1",
                    chunk_index=0,
                    total_chunks=2,
                ),
                _make_record(
                    "Goodbye world!",
                    [0.0, 1.0, 0.0],
                    document_id="doc-1",
                    chunk_index=1,
                    total_chunks=2,
                ),
            ],
        )

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
        )

        self.assertEqual(
            _dump_results(results),
            [
                {
                    "score": 1.0,
                    "document_id": "doc-1",
                    "chunk": {
                        "content": {
                            "type": "text",
                            "text": "Hello world!",
                            "id": AnyString(),
                        },
                        "source": "doc-1.txt",
                        "chunk_index": 0,
                        "total_chunks": 2,
                        "metadata": {},
                    },
                },
                {
                    "score": 0.0,
                    "document_id": "doc-1",
                    "chunk": {
                        "content": {
                            "type": "text",
                            "text": "Goodbye world!",
                            "id": AnyString(),
                        },
                        "source": "doc-1.txt",
                        "chunk_index": 1,
                        "total_chunks": 2,
                        "metadata": {},
                    },
                },
            ],
        )

    async def test_search_top_k(self) -> None:
        """top_k limits the number of returned results."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record("A", [1.0, 0.0, 0.0], document_id="doc-1"),
                _make_record("B", [0.9, 0.1, 0.0], document_id="doc-2"),
                _make_record("C", [0.0, 0.0, 1.0], document_id="doc-3"),
            ],
        )

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=1,
        )

        self.assertEqual(
            _dump_results(results),
            [
                {
                    "score": 1.0,
                    "document_id": "doc-1",
                    "chunk": {
                        "content": {
                            "type": "text",
                            "text": "A",
                            "id": AnyString(),
                        },
                        "source": "doc-1.txt",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "metadata": {},
                    },
                },
            ],
        )

    async def test_delete_by_document_id(self) -> None:
        """delete removes all records of one document only."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "doc1-chunk0",
                    [1.0, 0.0, 0.0],
                    document_id="doc-1",
                    chunk_index=0,
                    total_chunks=2,
                ),
                _make_record(
                    "doc1-chunk1",
                    [0.9, 0.1, 0.0],
                    document_id="doc-1",
                    chunk_index=1,
                    total_chunks=2,
                ),
                _make_record(
                    "doc2-chunk0",
                    [0.0, 1.0, 0.0],
                    document_id="doc-2",
                ),
            ],
        )

        await self.store.delete("kb-1", document_id="doc-1")

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
        )

        self.assertEqual(
            _dump_results(results),
            [
                {
                    "score": 0.0,
                    "document_id": "doc-2",
                    "chunk": {
                        "content": {
                            "type": "text",
                            "text": "doc2-chunk0",
                            "id": AnyString(),
                        },
                        "source": "doc-2.txt",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "metadata": {},
                    },
                },
            ],
        )

    async def test_insert_empty_records(self) -> None:
        """Inserting an empty record list is a no-op."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert("kb-1", [])

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
        )

        self.assertEqual(_dump_results(results), [])

    async def test_list_documents_aggregates_by_document_id(self) -> None:
        """list_documents groups chunks by document_id."""
        await self.store.create_collection("kb-1", dimensions=3)

        def _record_with_metadata(
            text: str,
            document_id: str,
            metadata: dict,
            chunk_index: int = 0,
            total_chunks: int = 1,
        ) -> VectorRecord:
            return VectorRecord(
                vector=[1.0, 0.0, 0.0],
                document_id=document_id,
                chunk=Chunk(
                    content=TextBlock(text=text),
                    source=metadata.get("filename", f"{document_id}.txt"),
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    metadata=metadata,
                ),
            )

        await self.store.insert(
            "kb-1",
            [
                _record_with_metadata(
                    "A",
                    "doc-1",
                    {"filename": "alpha.txt", "media_type": "text/plain"},
                    0,
                    2,
                ),
                _record_with_metadata(
                    "B",
                    "doc-1",
                    {"filename": "alpha.txt", "media_type": "text/plain"},
                    1,
                    2,
                ),
                _record_with_metadata(
                    "C",
                    "doc-2",
                    {"filename": "beta.md", "media_type": "text/markdown"},
                    0,
                    1,
                ),
            ],
        )

        summaries = await self.store.list_documents("kb-1")
        summaries_by_id = {s.document_id: s for s in summaries}

        self.assertEqual(set(summaries_by_id), {"doc-1", "doc-2"})
        self.assertEqual(summaries_by_id["doc-1"].chunk_count, 2)
        self.assertEqual(summaries_by_id["doc-1"].source, "alpha.txt")
        self.assertEqual(
            summaries_by_id["doc-1"].metadata,
            {"filename": "alpha.txt", "media_type": "text/plain"},
        )
        self.assertEqual(summaries_by_id["doc-2"].chunk_count, 1)
        self.assertEqual(summaries_by_id["doc-2"].source, "beta.md")

    async def test_search_metadata_filter(self) -> None:
        """search applies the metadata_filter as a payload predicate."""
        await self.store.create_collection("kb-1", dimensions=3)

        def _record(
            text: str,
            document_id: str,
            kb_scope: str,
        ) -> VectorRecord:
            return VectorRecord(
                vector=[1.0, 0.0, 0.0],
                document_id=document_id,
                chunk=Chunk(
                    content=TextBlock(text=text),
                    source=f"{document_id}.txt",
                    chunk_index=0,
                    total_chunks=1,
                    metadata={"kb_scope": kb_scope},
                ),
            )

        await self.store.insert(
            "kb-1",
            [
                _record("A", "doc-1", "kb-a"),
                _record("B", "doc-2", "kb-b"),
            ],
        )

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            metadata_filter={"kb_scope": "kb-a"},
        )
        self.assertEqual([r.document_id for r in results], ["doc-1"])

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
            metadata_filter={"kb_scope": "kb-b"},
        )
        self.assertEqual([r.document_id for r in results], ["doc-2"])

    async def test_euclidean_distance_orders_ascending(self) -> None:
        """With EUCLIDEAN distance, the closest vector (lowest L2) is first."""
        store = AnalyticDBMySQLStore.__new__(AnalyticDBMySQLStore)
        store._database = "rag"
        store._distance = "EUCLIDEAN"
        store._conn_kwargs = {}
        store._conn = _FakeConnection()

        await store.create_collection("kb-1", dimensions=3)
        await store.insert(
            "kb-1",
            [
                _make_record("near", [1.0, 0.0, 0.0], document_id="doc-1"),
                _make_record("far", [0.0, 0.0, 1.0], document_id="doc-2"),
            ],
        )

        results = await store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
        )

        self.assertEqual([r.document_id for r in results], ["doc-1", "doc-2"])
        self.assertAlmostEqual(results[0].score, 0.0)
        self.assertAlmostEqual(results[1].score, math.sqrt(2.0))
        store.close()
