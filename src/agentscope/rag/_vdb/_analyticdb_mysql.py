# -*- coding: utf-8 -*-
"""The AlibabaCloud AnalyticDB MySQL vector store implementation."""
import json
import re
from typing import Any, Literal, TYPE_CHECKING

from ._vector_store import (
    DocumentSummary,
    VectorRecord,
    VectorSearchResult,
    VectorStoreBase,
)
from .._document import Chunk

if TYPE_CHECKING:
    from pymysql.connections import Connection
else:
    Connection = "pymysql.connections.Connection"


# Per metric: the ANN index ``distancemeasure`` used in DDL, then the
# query-time scoring function.
_METRIC_SQL = {
    "COSINE": ("CosineSimilarity", "cosine_similarity"),
    "EUCLIDEAN": ("SquaredL2", "l2_distance"),
}

# 64 is the MySQL identifier limit.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class AnalyticDBMySQLStore(VectorStoreBase):
    """The AlibabaCloud AnalyticDB MySQL vector store implementation.

    A single instance owns one ``pymysql`` connection to a database, and
    each knowledge base maps to one table inside it.  Every row persists
    the ``document_id`` and the serialized :class:`Chunk`, so
    :meth:`delete` can remove a document as a unit and
    :meth:`list_documents` can aggregate its chunks.

    .. note:: The ``pymysql`` package is required. Install it with
        ``pip install "agentscope[vdb-analyticdb]"``.  The driver is
        synchronous and its connection is not thread-safe, so keep one
        store to a single thread at a time.

    .. code-block:: python

        store = AnalyticDBMySQLStore(
            host="localhost",
            port=3306,
            user="alice",
            password="secret",
            database="rag",
        )

        async with store:
            await store.create_collection("kb-1", dimensions=768)
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        distance: Literal["COSINE", "EUCLIDEAN"] = "COSINE",
        connection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the AnalyticDB MySQL vector store.

        Args:
            host (`str`):
                The hostname of the AnalyticDB MySQL server.
            port (`int`):
                The port number of the AnalyticDB MySQL server.
            user (`str`):
                The username for authentication.
            password (`str`):
                The password for authentication.
            database (`str`):
                The database name; each collection is a table inside it.
            distance (`Literal["COSINE", "EUCLIDEAN"]`, defaults to \
                ``"COSINE"``):
                The similarity metric.  ``"COSINE"`` scores rise with
                similarity, ``"EUCLIDEAN"`` scores fall with it.
            connection_kwargs (`dict[str, Any] | None`, optional):
                Other keyword arguments for the MySQL connector.
                Example: ``{"ssl_ca": "/path/to/ca.pem", \
                "charset": "utf8mb4"}``.  ``autocommit`` defaults to
                ``True``.

        Raises:
            `ImportError`: If the ``pymysql`` package is not installed.
            `ValueError`: If ``distance`` or ``database`` is invalid.
        """
        if distance not in _METRIC_SQL:
            raise ValueError(
                f"Unsupported distance metric: {distance!r}. AnalyticDB "
                f"MySQL only supports 'COSINE' and 'EUCLIDEAN'.",
            )

        try:
            import pymysql
        except ImportError as e:
            raise ImportError(
                "Could not import pymysql python package. "
                "Please install it with `pip install "
                "agentscope[vdb-analyticdb]`.",
            ) from e

        self._database = self._safe_identifier(database, "database name")
        self._distance = distance
        self._conn_kwargs = {"autocommit": True, **(connection_kwargs or {})}
        self._conn: "Connection | None" = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            **self._conn_kwargs,
        )

    def _require_conn(self) -> "Connection":
        """Return the live connection, or raise if :meth:`close` has run."""
        if self._conn is None:
            raise RuntimeError("The AnalyticDB MySQL store is closed.")
        return self._conn

    def get_client(self) -> "Connection":
        """Return the underlying MySQL connection."""
        return self._require_conn()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the async context — close the underlying connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_sql(
        self,
        sql: str,
        data: dict[str, Any] | list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute one SQL statement and return its rows as dicts.

        Args:
            sql (`str`):
                The SQL statement.  DDL / DML or a SELECT query.
            data (`dict[str, Any] | list[Any] | None`, optional):
                Parameters for a parameterized query, or a list of
                parameter dicts / values for ``executemany``.

        Returns:
            `list[dict[str, Any]]`:
                SELECT rows keyed by column name; empty for a statement
                that returns no result set.

        Raises:
            `RuntimeError`: If the store has been closed.
        """
        conn = self._require_conn()
        conn.ping(reconnect=True)
        cursor = conn.cursor()
        try:
            if data is None:
                cursor.execute(sql)
            elif (
                isinstance(data, list)
                and len(data) > 0
                and isinstance(data[0], dict)
            ):
                cursor.executemany(sql, data)
            else:
                cursor.execute(sql, data)

            if cursor.description is None:
                return []

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def _safe_identifier(name: str, kind: str) -> str:
        """Return ``name`` if it is safe to splice into SQL."""
        if not _SAFE_IDENTIFIER_RE.match(name):
            raise ValueError(
                f"Invalid {kind} {name!r}: use 1-64 characters from "
                f"[A-Za-z0-9_-], starting with a letter or digit.",
            )
        return name

    def _qualified(self, collection: str) -> str:
        """Return the backtick-qualified ``database.collection`` reference.

        Raises:
            `ValueError`: If ``collection`` is not a safe identifier.
        """
        self._safe_identifier(collection, "collection name")
        return f"`{self._database}`.`{collection}`"

    @staticmethod
    def _metadata_conditions(
        metadata_filter: dict[str, Any] | None,
    ) -> tuple[list[str], list[Any]]:
        """Translate a flat ``{key: value}`` filter into one ``JSON_EXTRACT``
        predicate per key against ``chunk.metadata.<key>``, plus the values
        to bind them with, in matching order.

        Raises:
            `ValueError`: If a key is not a safe JSON path segment.
        """
        conditions: list[str] = []
        values: list[Any] = []
        for key, value in (metadata_filter or {}).items():
            AnalyticDBMySQLStore._safe_identifier(key, "metadata key")
            conditions.append(
                f"JSON_EXTRACT(chunk, '$.metadata.{key}') = CAST(%s AS JSON)",
            )
            values.append(json.dumps(value))
        return conditions, values

    @staticmethod
    def _where_clause(conditions: list[str]) -> str:
        """AND the predicates into a ``WHERE`` clause, or ``""`` if none."""
        if not conditions:
            return ""
        return "WHERE " + " AND ".join(conditions)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        dimensions: int,
    ) -> None:
        """Create a new table (collection), no-op if it already exists.

        Args:
            name (`str`):
                The collection name. Typically, the knowledge base ID.
            dimensions (`int`):
                The fixed vector dimensionality for this collection.

        Raises:
            `ValueError`: If ``name`` or ``dimensions`` is invalid.
        """
        if dimensions <= 0:
            raise ValueError(
                f"Vector dimensions must be positive, got {dimensions}.",
            )

        measure, _ = _METRIC_SQL[self._distance]
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._qualified(name)} (
            id VARCHAR(255) PRIMARY KEY,
            embedding ARRAY<FLOAT>({dimensions}) NOT NULL,
            document_id VARCHAR(255) NOT NULL,
            chunk JSON NOT NULL,
            ANN INDEX idx_vector(embedding) distancemeasure={measure}
        ) ENGINE='XUANWU_V2'
        """
        self._execute_sql(create_sql)

    async def delete_collection(self, name: str) -> None:
        """Delete a collection (table) and all its data.

        Args:
            name (`str`):
                The collection name to delete.
        """
        self._execute_sql(f"DROP TABLE IF EXISTS {self._qualified(name)}")

    async def has_collection(self, name: str) -> bool:
        """Check whether a collection (table) exists.

        Args:
            name (`str`):
                The collection name to check.

        Returns:
            `bool`: ``True`` if the collection exists.

        Raises:
            `ValueError`: If ``name`` is not a safe identifier.
        """
        self._safe_identifier(name, "collection name")
        pattern = name.replace("_", "\\_")
        return bool(self._execute_sql(f"SHOW TABLES LIKE '{pattern}'"))

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    async def insert(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Insert records into a collection.

        Rows are keyed on ``document_id`` + ``chunk_index`` and written with
        ``REPLACE INTO``, so re-indexing a document replaces its rows
        instead of appending duplicates.

        Args:
            collection (`str`):
                The target collection name.
            records (`list[VectorRecord]`):
                The records to insert (each carrying a
                :class:`Chunk` and its embedding vector).

        Raises:
            `ValueError`: If ``collection`` is not a safe identifier.
        """
        if not records:
            return

        insert_sql = (
            f"REPLACE INTO {self._qualified(collection)} "
            "(id, embedding, document_id, chunk) VALUES "
            "(%(id)s, %(embedding)s, %(document_id)s, %(chunk)s)"
        )
        data = [
            {
                "id": f"{record.document_id}:{record.chunk.chunk_index}",
                "embedding": json.dumps(record.vector),
                "document_id": record.document_id,
                "chunk": record.chunk.model_dump_json(),
            }
            for record in records
        ]
        self._execute_sql(insert_sql, data)

    async def delete(
        self,
        collection: str,
        document_id: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        """Delete all records belonging to one source document.

        Matches the ``document_id`` column written by :meth:`insert`, plus
        every ``chunk.metadata`` predicate when ``metadata_filter`` is
        given, so a scoped knowledge base never deletes outside its scope.

        Args:
            collection (`str`):
                The target collection name.
            document_id (`str`):
                The source document ID whose records should be
                removed.
            metadata_filter (`dict[str, Any] | None`, optional):
                If provided, restrict deletion to records whose
                ``chunk.metadata`` matches every ``key == value`` pair
                in this dict, in addition to the document ID.

        Raises:
            `ValueError`: If ``collection`` or a metadata key is invalid.
        """
        conditions, values = self._metadata_conditions(metadata_filter)
        delete_sql = (
            f"DELETE FROM {self._qualified(collection)} "
            f"{self._where_clause(['document_id = %s', *conditions])}"
        )
        self._execute_sql(delete_sql, [document_id, *values])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Find the most similar records to a query vector.

        Args:
            collection (`str`):
                The collection to search.
            query_vector (`list[float]`):
                The query embedding vector.
            top_k (`int`, defaults to ``5``):
                Maximum number of results to return.
            metadata_filter (`dict[str, Any] | None`, optional):
                If provided, restrict the search to records whose
                ``chunk.metadata`` matches every ``key == value`` pair
                in this dict.

        Returns:
            `list[VectorSearchResult]`:
                Results ordered by descending score; an ``EUCLIDEAN``
                distance is negated to keep that ordering.

        Raises:
            `ValueError`: If ``top_k`` is negative or ``collection`` is
                not a safe identifier.
        """
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k!r}.")

        # ``l2_distance`` is lower-is-better, so its sign is flipped below.
        higher_is_better = self._distance == "COSINE"
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        _, distance_func = _METRIC_SQL[self._distance]
        conditions, values = self._metadata_conditions(metadata_filter)
        where_clause = self._where_clause(conditions)

        search_sql = f"""
        SELECT
            id,
            document_id,
            chunk,
            {distance_func}(embedding, '{vector_str}') AS distance
        FROM {self._qualified(collection)}
        {where_clause}
        ORDER BY distance {'DESC' if higher_is_better else 'ASC'}
        LIMIT {top_k}
        """
        rows = self._execute_sql(search_sql, values or None)
        sign = 1 if higher_is_better else -1

        return [
            VectorSearchResult(
                score=sign * float(row["distance"]),
                document_id=row["document_id"],
                chunk=Chunk.model_validate_json(row["chunk"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Document listing
    # ------------------------------------------------------------------

    async def list_documents(
        self,
        collection: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentSummary]:
        """List all distinct source documents indexed in a collection.

        Scans the rows (vectors are not needed) and aggregates them by
        ``document_id``; each document's first chunk supplies the
        ``source`` filename and the summary ``metadata``.

        Args:
            collection (`str`):
                The target collection name.
            metadata_filter (`dict[str, Any] | None`, optional):
                If provided, restrict aggregation to records whose
                ``chunk.metadata`` matches every ``key == value`` pair.

        Returns:
            `list[DocumentSummary]`:
                One summary per distinct ``document_id``, in
                unspecified order.

        Raises:
            `ValueError`: If ``collection`` or a metadata key is invalid.
        """
        conditions, values = self._metadata_conditions(metadata_filter)
        list_sql = (
            f"SELECT document_id, chunk "
            f"FROM {self._qualified(collection)} "
            f"{self._where_clause(conditions)}"
        )
        rows = self._execute_sql(list_sql, values or None)

        summaries: dict[str, DocumentSummary] = {}
        for row in rows:
            doc_id = row["document_id"]
            summary = summaries.get(doc_id)
            if summary is None:
                chunk = Chunk.model_validate_json(row["chunk"])
                summaries[doc_id] = DocumentSummary(
                    document_id=doc_id,
                    source=chunk.source,
                    chunk_count=1,
                    metadata=dict(chunk.metadata),
                )
            else:
                summary.chunk_count += 1
        return list(summaries.values())

    # ------------------------------------------------------------------
    # Chunk listing
    # ------------------------------------------------------------------

    async def list_chunks(
        self,
        collection: str,
        document_id: str,
        *,
        offset: int = 0,
        limit: int = 30,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """List one document's chunks ordered by ``chunk_index``.

        Pushes the ``offset <= chunk_index < offset + limit`` window into
        the ``WHERE`` clause instead of sorting and skipping, so a page
        never scans the whole document.  The primary key
        (``document_id`` + ``chunk_index``) keeps those rows unique, so
        the window alone bounds the page.

        Args:
            collection (`str`):
                The target collection name.
            document_id (`str`):
                The source document whose chunks should be listed.
            offset (`int`, defaults to ``0``):
                Number of leading chunks to skip.
            limit (`int`, defaults to ``30``):
                Maximum number of chunks to return.
            metadata_filter (`dict[str, Any] | None`, optional):
                Extra ``chunk.metadata`` equality constraints.

        Returns:
            `list[Chunk]`:
                At most ``limit`` chunks, ``chunk_index`` ascending.

        Raises:
            `ValueError`: If ``collection`` or a metadata key is invalid.
        """
        if limit <= 0:
            return []

        conditions, values = self._metadata_conditions(metadata_filter)
        where_clause = self._where_clause(
            [
                "document_id = %s",
                "JSON_EXTRACT(chunk, '$.chunk_index') >= %s",
                "JSON_EXTRACT(chunk, '$.chunk_index') < %s",
                *conditions,
            ],
        )
        list_sql = f"""
        SELECT chunk
        FROM {self._qualified(collection)}
        {where_clause}
        ORDER BY JSON_EXTRACT(chunk, '$.chunk_index')
        """
        rows = self._execute_sql(
            list_sql,
            [document_id, offset, offset + limit, *values],
        )
        return [Chunk.model_validate_json(row["chunk"]) for row in rows]
