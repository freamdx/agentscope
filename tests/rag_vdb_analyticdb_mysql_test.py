# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for the AnalyticDBMySQLStore class.

AnalyticDB for MySQL is a hosted vector database, so CI cannot spin one
up the way the Qdrant tests use an in-process instance.  Instead a fake
pymysql connection interprets the SQL the store emits and simulates the
vector semantics (``ARRAY<FLOAT>`` columns, ``cosine_similarity`` /
``l2_distance``, ``JSON_EXTRACT`` predicates) in Python, which keeps the
store's real SQL generation and result parsing under test.
"""
import json
import math
import re
from contextlib import AsyncExitStack
from typing import Any
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import patch

from utils import AnyString

from agentscope.message import TextBlock
from agentscope.rag import (
    AnalyticDBMySQLStore,
    Chunk,
    VectorRecord,
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
        return sql % tuple(params)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    @staticmethod
    def _l2(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _parse_metadata_filter(self, sql: str) -> dict[str, Any]:
        """Extract ``{key: value}`` pairs from the JSON_EXTRACT WHERE."""
        return {
            match.group(1): json.loads(match.group(2))
            for match in re.finditer(
                r"JSON_EXTRACT\(chunk, '\$\.metadata\.([^']+)'\)\s*=\s*"
                r"CAST\((.+?) AS JSON\)",
                sql,
            )
        }

    @staticmethod
    def _metadata_matches(row: dict, meta_filter: dict[str, Any]) -> bool:
        """Return whether one stored row satisfies the filter."""
        metadata = json.loads(row["chunk"]).get("metadata", {})
        return all(metadata.get(k) == v for k, v in meta_filter.items())

    @staticmethod
    def _chunk_index(row: dict) -> int:
        """Read back the ``chunk_index`` a row was inserted with."""
        return json.loads(row["chunk"])["chunk_index"]

    # -- execute -------------------------------------------------------
    def execute(self, sql: str, params: Any = None) -> None:
        """Dispatch a single SQL statement, populating ``description`` /
        ``_rows`` for SELECTs, or mutating the in-memory tables otherwise.
        """
        # Record the statement as the driver would receive it, before the
        # ``%s`` placeholders are filled in.
        self._conn._statements.append(sql)
        sql = self._substitute(sql, params)
        self.description = None
        self._rows = []
        tables = self._conn._tables
        lowered = sql.strip()

        # CREATE TABLE IF NOT EXISTS `db`.`name` (...)
        if lowered.startswith("CREATE TABLE"):
            tables.setdefault(self._qualified_name(sql), {"rows": []})
            return

        # DROP TABLE IF EXISTS `db`.`name`
        if lowered.startswith("DROP TABLE"):
            tables.pop(self._qualified_name(sql), None)
            return

        # SHOW TABLES LIKE 'name'
        if lowered.startswith("SHOW TABLES"):
            match = re.search(r"LIKE '([^']+)'", sql)
            assert match is not None
            # Undo the store's ``\_`` escaping; this fake matches exactly.
            name = match.group(1).replace("\\_", "_")
            self.description = [("Tables_in_db",)]
            self._rows = [(name,)] if name in tables else []
            return

        # REPLACE INTO `db`.`name` (cols) VALUES (%(id)s, ...)
        if lowered.startswith("REPLACE INTO"):
            assert isinstance(params, dict)
            row = {
                "id": params["id"],
                "embedding": json.loads(params["embedding"]),
                "document_id": params["document_id"],
                "chunk": params["chunk"],
            }
            # REPLACE semantics: upsert by id.
            rows = tables[self._qualified_name(sql)]["rows"]
            for i, existing in enumerate(rows):
                if existing["id"] == row["id"]:
                    rows[i] = row
                    break
            else:
                rows.append(row)
            return

        # DELETE FROM `db`.`name` WHERE document_id = %s
        # [AND JSON_EXTRACT(chunk, '$.metadata.<key>') = CAST(...)]
        if lowered.startswith("DELETE FROM"):
            name = self._qualified_name(sql)
            doc_id = params[0]
            meta_filter = self._parse_metadata_filter(sql)
            # Keep every row that is not selected by the delete predicate.
            tables[name]["rows"] = [
                row
                for row in tables[name]["rows"]
                if not (
                    row["document_id"] == doc_id
                    and self._metadata_matches(row, meta_filter)
                )
            ]
            return

        # SELECT ... — search, list_chunks or list_documents
        if lowered.startswith("SELECT"):
            name = self._qualified_name(sql)
            rows = tables.get(name, {}).get("rows", [])
            meta_filter = self._parse_metadata_filter(sql)

            filtered = [
                row for row in rows if self._metadata_matches(row, meta_filter)
            ]

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
                    (metric(row["embedding"], query_vec), row)
                    for row in filtered
                ]
                scored.sort(
                    key=lambda x: x[0],
                    reverse=func == "cosine_similarity",
                )
                limit_match = re.search(r"LIMIT (\d+)", sql)
                assert limit_match is not None
                scored = scored[: int(limit_match.group(1))]
                self.description = [
                    ("id",),
                    ("document_id",),
                    ("chunk",),
                    ("distance",),
                ]
                self._rows = [
                    (row["id"], row["document_id"], row["chunk"], score)
                    for score, row in scored
                ]
                return

            # --- list_chunks ---
            if "$.chunk_index" in sql:
                doc_id, start, end = params[:3]
                page = [
                    row
                    for row in filtered
                    if row["document_id"] == doc_id
                    and start <= self._chunk_index(row) < end
                ]
                page.sort(key=self._chunk_index)
                self.description = [("chunk",)]
                self._rows = [(row["chunk"],) for row in page]
                return

            # --- list_documents ---
            self.description = [("document_id",), ("chunk",)]
            self._rows = [
                (row["document_id"], row["chunk"]) for row in filtered
            ]
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
        self._statements: list[str] = []

    def cursor(self) -> _FakeCursor:
        """Return a fresh cursor bound to this connection."""
        return _FakeCursor(self)

    def ping(self, reconnect: bool = False) -> None:
        """No-op — the fake never drops, ``reconnect`` is ignored."""

    def rollback(self) -> None:
        """No-op — the fake has no transactions."""

    def close(self) -> None:
        """No-op — there is nothing to close."""


# ======================================================================
# Test helpers
# =======================================================================


def _make_store(distance: str = "COSINE") -> AnalyticDBMySQLStore:
    """Build an :class:`AnalyticDBMySQLStore` backed by a fake connection.

    Bypasses ``__init__``, which would open a real connection to the
    hosted cluster.
    """
    store = AnalyticDBMySQLStore.__new__(AnalyticDBMySQLStore)
    store._database = "rag"
    store._distance = distance
    store._conn_kwargs = {}
    store._conn = _FakeConnection()
    return store


def _connect_kwargs(**store_kwargs: Any) -> dict[str, Any]:
    """Build a real store with the driver mocked and return the keyword
    arguments it passed to ``pymysql.connect``."""
    with patch("pymysql.connect") as mock_connect:
        AnalyticDBMySQLStore(
            host="localhost",
            port=3306,
            user="alice",
            password="secret",
            database="rag",
            **store_kwargs,
        )
    return dict(mock_connect.call_args.kwargs)


def _make_record(
    text: str,
    vector: list[float],
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
    metadata: dict[str, Any] | None = None,
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
            metadata=metadata or {},
        ),
    )


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
            [result.model_dump() for result in results],
            [
                {
                    "score": 1.0,
                    "document_id": "doc-1",
                    "chunk": {
                        "content": {
                            "type": "text",
                            "text": "Hello world!",
                            "id": AnyString(),
                            "created_at": AnyString(),
                            "finished_at": None,
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
                            "created_at": AnyString(),
                            "finished_at": None,
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

        self.assertEqual([r.document_id for r in results], ["doc-1"])

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

        self.assertEqual([r.document_id for r in results], ["doc-2"])

    async def test_insert_empty_records(self) -> None:
        """Inserting an empty record list is a no-op."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert("kb-1", [])

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
        )

        self.assertEqual(results, [])

    async def test_list_documents_aggregates_by_document_id(self) -> None:
        """list_documents groups chunks by document_id."""
        await self.store.create_collection("kb-1", dimensions=3)
        alpha = {"filename": "alpha.txt", "media_type": "text/plain"}
        beta = {"filename": "beta.md", "media_type": "text/markdown"}

        await self.store.insert(
            "kb-1",
            [
                _make_record("A", [1.0, 0.0, 0.0], "doc-1", 0, 2, alpha),
                _make_record("B", [1.0, 0.0, 0.0], "doc-1", 1, 2, alpha),
                _make_record("C", [1.0, 0.0, 0.0], "doc-2", 0, 1, beta),
            ],
        )

        summaries = await self.store.list_documents("kb-1")
        summaries_by_id = {s.document_id: s for s in summaries}

        self.assertEqual(set(summaries_by_id), {"doc-1", "doc-2"})
        self.assertEqual(summaries_by_id["doc-1"].chunk_count, 2)
        self.assertEqual(summaries_by_id["doc-1"].source, "doc-1.txt")
        self.assertEqual(summaries_by_id["doc-1"].metadata, alpha)
        self.assertEqual(summaries_by_id["doc-2"].chunk_count, 1)
        self.assertEqual(summaries_by_id["doc-2"].source, "doc-2.txt")

    async def test_list_chunks_pages_by_chunk_index(self) -> None:
        """list_chunks pushes the index window down and pages in order."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    f"chunk-{index}",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    index,
                    5,
                    {"kb_scope": "kb-a" if index < 3 else "kb-b"},
                )
                for index in range(5)
            ],
        )

        cases: list[tuple[dict[str, Any], list[int]]] = [
            ({}, [0, 1, 2, 3, 4]),
            ({"offset": 1, "limit": 2}, [1, 2]),
            ({"offset": 3, "limit": 2}, [3, 4]),
            ({"offset": 5}, []),
            ({"limit": 0}, []),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                chunks = await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    **kwargs,
                )
                self.assertEqual([c.chunk_index for c in chunks], expected)

        chunks = await self.store.list_chunks(
            "kb-1",
            "doc-1",
            offset=1,
            limit=2,
            metadata_filter={"kb_scope": "kb-a"},
        )
        self.assertEqual([c.chunk_index for c in chunks], [1, 2])

        # The window and the ordering live in the statement, with the
        # values bound as parameters rather than spliced in.
        statement = self.store._conn._statements[-1]
        for fragment in [
            "document_id = %s",
            "JSON_EXTRACT(chunk, '$.chunk_index') >= %s",
            "JSON_EXTRACT(chunk, '$.chunk_index') < %s",
            "ORDER BY JSON_EXTRACT(chunk, '$.chunk_index')",
        ]:
            self.assertIn(fragment, statement)

    async def test_search_metadata_filter(self) -> None:
        """search applies the metadata_filter as a payload predicate."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "A",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    metadata={"kb_scope": "kb-a"},
                ),
                _make_record(
                    "B",
                    [1.0, 0.0, 0.0],
                    "doc-2",
                    metadata={"kb_scope": "kb-b"},
                ),
            ],
        )

        for scope, expected in [("kb-a", "doc-1"), ("kb-b", "doc-2")]:
            with self.subTest(scope=scope):
                results = await self.store.search(
                    "kb-1",
                    query_vector=[1.0, 0.0, 0.0],
                    top_k=5,
                    metadata_filter={"kb_scope": scope},
                )
                self.assertEqual([r.document_id for r in results], [expected])

        # The value is bound as a parameter, not spliced into the statement.
        self.assertIn("CAST(%s AS JSON)", self.store._conn._statements[-1])

    async def test_euclidean_distance_orders_ascending(self) -> None:
        """With EUCLIDEAN distance, the closest vector (lowest L2) is first."""
        store = _make_store("EUCLIDEAN")

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
        # The distance is negated to keep the score higher-is-better.
        self.assertAlmostEqual(results[0].score, 0.0)
        self.assertAlmostEqual(results[1].score, -math.sqrt(2.0))
        self.assertGreater(results[0].score, results[1].score)
        store.close()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def test_construction(self) -> None:
        """``__init__`` pins autocommit and refuses an unknown metric."""
        cases: list[tuple[dict[str, Any], bool]] = [
            ({}, True),
            ({"connection_kwargs": {"autocommit": False}}, False),
        ]
        for store_kwargs, expected in cases:
            with self.subTest(**store_kwargs):
                kwargs = _connect_kwargs(**store_kwargs)
                self.assertEqual(kwargs.get("autocommit"), expected)

        with self.assertRaises(ValueError):
            _connect_kwargs(distance="INNER_PRODUCT")

    async def test_closed_store_rejects_use(self) -> None:
        """After ``close`` the store raises instead of handing out None."""
        self.store.close()

        with self.assertRaises(RuntimeError):
            self.store.get_client()

        with self.assertRaises(RuntimeError):
            await self.store.delete_collection("kb-1")

    # ------------------------------------------------------------------
    # Collection DDL
    # ------------------------------------------------------------------

    async def test_create_collection_ddl(self) -> None:
        """One re-runnable CREATE carries the ANN index of the metric."""
        cases = [("COSINE", "CosineSimilarity"), ("EUCLIDEAN", "SquaredL2")]
        for distance, measure in cases:
            with self.subTest(distance=distance):
                store = _make_store(distance)
                await store.create_collection("kb-1", dimensions=3)
                await store.create_collection("kb-1", dimensions=3)

                statements = store._conn._statements
                # No follow-up ALTER: the second call adds no index work.
                self.assertEqual(len(statements), 2)
                for statement in statements:
                    self.assertIn("CREATE TABLE IF NOT EXISTS", statement)
                self.assertIn(
                    f"ANN INDEX idx_vector(embedding) "
                    f"distancemeasure={measure}",
                    statements[0],
                )
                self.assertIn("ENGINE='XUANWU_V2'", statements[0])
                self.assertEqual(await store.has_collection("kb-1"), True)
                store.close()

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    async def test_reinsert_same_document_replaces_rows(self) -> None:
        """Re-indexing overwrites a document's rows, never duplicates."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [_make_record("v1", [1.0, 0.0, 0.0], document_id="doc-1")],
        )
        await self.store.insert(
            "kb-1",
            [_make_record("v2", [1.0, 0.0, 0.0], document_id="doc-1")],
        )

        summaries = await self.store.list_documents("kb-1")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].chunk_count, 1)

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
        )
        self.assertEqual(len(results), 1)
        text = results[0].chunk.model_dump()["content"]["text"]
        self.assertEqual(text, "v2")

    async def test_delete_respects_metadata_filter(self) -> None:
        """A scoped delete only removes rows inside the scope."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    scope,
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    index,
                    2,
                    {"kb_scope": scope},
                )
                for index, scope in enumerate(["kb-a", "kb-b"])
            ],
        )

        await self.store.delete(
            "kb-1",
            document_id="doc-1",
            metadata_filter={"kb_scope": "kb-a"},
        )

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=5,
        )
        self.assertEqual(
            [r.chunk.metadata["kb_scope"] for r in results],
            ["kb-b"],
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    async def test_rejects_invalid_arguments(self) -> None:
        """Bad names, sizes and keys fail before reaching the cluster."""
        await self.store.create_collection("kb-1", dimensions=3)
        record = _make_record("A", [1.0, 0.0, 0.0], document_id="doc-1")

        cases = {
            "collection in DDL": lambda: self.store.create_collection(
                "kb-1`; DROP TABLE x",
                3,
            ),
            "collection in DML": lambda: self.store.insert("kb 1", [record]),
            "collection in SHOW": lambda: self.store.has_collection("nope!"),
            "metadata key in search": lambda: self.store.search(
                "kb-1",
                query_vector=[1.0, 0.0, 0.0],
                metadata_filter={"evil' = 1 OR 1=1 -- ": "x"},
            ),
            "metadata key in delete": lambda: self.store.delete(
                "kb-1",
                document_id="doc-1",
                metadata_filter={"a.b": "x"},
            ),
            "non-positive dimensions": lambda: self.store.create_collection(
                "kb-1",
                dimensions=0,
            ),
            "negative top_k": lambda: self.store.search(
                "kb-1",
                query_vector=[1.0, 0.0, 0.0],
                top_k=-1,
            ),
        }
        for label, call in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    await call()

    async def test_has_collection_escapes_like_wildcards(self) -> None:
        """Generated ``kb_<id>`` names must not read as LIKE patterns."""
        await self.store.has_collection("kb_1")

        statement = self.store._conn._statements[-1]
        self.assertIn(r"SHOW TABLES LIKE 'kb\_1'", statement)
