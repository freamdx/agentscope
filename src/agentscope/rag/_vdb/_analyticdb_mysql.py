# -*- coding: utf-8 -*-
"""The AlibabaCloud AnalyticDB MySQL vector store implementation.

Each knowledge base maps to one AnalyticDB MySQL table.  Every row
stores the owning ``document_id`` plus the serialized
:class:`~agentscope.rag.Chunk` (as a JSON document), which is
reconstructed on retrieval.

.. note:: The ``pymysql`` package is required. Install it with
    ``pip install pymysql``.
"""
import json
import uuid
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


class AnalyticDBMySQLStore(VectorStoreBase):
    """The AlibabaCloud AnalyticDB MySQL vector store implementation.

    A single instance owns one ``pymysql`` connection to a database.
    Each knowledge base maps to one table inside that database; the
    table name is the collection name.  Every row persists the
    ``document_id`` and the serialized :class:`Chunk`, so that
    :meth:`delete` can remove all records of one document as a unit
    and :meth:`list_documents` can aggregate them.

    .. note:: The ``pymysql`` package is required. Install it with
        ``pip install pymysql``.

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
                The database name to use.  Each collection is created
                as a table inside this database.
            distance (`Literal["COSINE", "EUCLIDEAN"]`, defaults to \
                ``"COSINE"``):
                The distance metric to use for similarity search.
                ``"COSINE"`` (cosine similarity, higher = more similar)
                or ``"EUCLIDEAN"`` (L2 distance, lower = more similar).
            connection_kwargs (`dict[str, Any] | None`, optional):
                Other keyword arguments for the MySQL connector.
                Example: ``{"ssl_ca": "/path/to/ca.pem", \
                "charset": "utf8mb4"}``
        """
        try:
            import pymysql
        except ImportError as e:
            raise ImportError(
                "Could not import pymysql python package. "
                "Please install it with `pip install pymysql`.",
            ) from e

        self._database = database
        self._distance = distance
        self._conn_kwargs = connection_kwargs or {}
        self._conn: "Connection | None" = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            **self._conn_kwargs,
        )

    def get_client(self) -> "Connection":
        """Return the underlying MySQL connection.

        Returns:
            `Connection`:
                The shared ``pymysql`` connection.
        """
        return self._conn  # type: ignore[return-value]

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
                The result rows of a SELECT as a list of dicts keyed
                by column name; an empty list for non-query statements.
        """
        cursor = self._conn.cursor()
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

            self._conn.commit()
            if cursor.description is None:
                return []

            columns = cursor.description
            result = []
            for value in cursor.fetchall():
                r = {}
                for idx, datum in enumerate(value):
                    r[columns[idx][0]] = datum
                result.append(r)
            return result
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _qualified(self, collection: str) -> str:
        """Return the backtick-qualified ``database.collection`` name."""
        return f"`{self._database}`.`{collection}`"

    def _get_distance_function(self) -> str:
        """Return the SQL vector distance function name."""
        if self._distance == "COSINE":
            return "cosine_similarity"
        if self._distance == "EUCLIDEAN":
            return "l2_distance"
        raise ValueError(
            f"Unsupported distance metric: {self._distance}. "
            f"AnalyticDB MySQL only supports 'COSINE' and 'EUCLIDEAN'.",
        )

    def _build_metadata_where(
        self,
        metadata_filter: dict[str, Any] | None,
    ) -> str:
        """Translate a flat ``{key: value}`` filter into a WHERE clause.

        Each ``key`` is matched against the corresponding nested path
        ``chunk.metadata.<key>`` written by :meth:`insert`, using
        ``JSON_EXTRACT`` so the comparison is type-agnostic.  Returns an
        empty string when ``metadata_filter`` is empty so that callers
        can splice the clause directly into any query.
        """
        if not metadata_filter:
            return ""
        conditions = [
            f"JSON_EXTRACT(chunk, '$.metadata.{key}') = "
            f"CAST({json.dumps(value)} AS JSON)"
            for key, value in metadata_filter.items()
        ]
        return "WHERE " + " AND ".join(conditions)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        dimensions: int,
    ) -> None:
        """Create a new table (collection).

        No-op if the table already exists.

        Args:
            name (`str`):
                The collection name. Typically, the knowledge base ID.
            dimensions (`int`):
                The fixed vector dimensionality for this collection.
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._qualified(name)} (
            id VARCHAR(255) PRIMARY KEY,
            embedding ARRAY<FLOAT>({dimensions}) NOT NULL,
            document_id VARCHAR(255) NOT NULL,
            chunk JSON NOT NULL,
            ANN INDEX idx_vector_embedding(embedding)
        )
        """
        self._execute_sql(create_table_sql)

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
        """
        rows = self._execute_sql(f"SHOW TABLES LIKE '{name}'")
        return len(rows) > 0

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    async def insert(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Insert records into a collection.

        Each row persists the :attr:`VectorRecord.document_id` under
        the ``document_id`` column and the serialized
        :class:`Chunk` under the ``chunk`` column, so that
        :meth:`delete` can remove all records of one document and
        :meth:`search` / :meth:`list_documents` can reconstruct it.

        Uses ``REPLACE INTO`` so re-inserting a record with the same
        ``id`` overwrites the previous row instead of raising.

        Args:
            collection (`str`):
                The target collection name.
            records (`list[VectorRecord]`):
                The records to insert (each carrying a
                :class:`Chunk` and its embedding vector).
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
                "id": str(uuid.uuid4()),
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
    ) -> None:
        """Delete all records belonging to one source document.

        Matches the ``document_id`` column written by :meth:`insert`.

        Args:
            collection (`str`):
                The target collection name.
            document_id (`str`):
                The source document ID whose records should be
                removed.
        """
        delete_sql = (
            f"DELETE FROM {self._qualified(collection)} "
            "WHERE document_id = %s"
        )
        self._execute_sql(delete_sql, [document_id])

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
                in this dict (translated into a ``JSON_EXTRACT`` based
                predicate).

        Returns:
            `list[VectorSearchResult]`:
                Results ordered by descending similarity score
                (cosine) or ascending distance (euclidean).
        """
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        distance_func = self._get_distance_function()
        order_by = "DESC" if self._distance == "COSINE" else "ASC"
        where_clause = self._build_metadata_where(metadata_filter)

        search_sql = f"""
        SELECT
            id,
            document_id,
            chunk,
            {distance_func}(embedding, '{vector_str}') AS distance
        FROM {self._qualified(collection)}
        {where_clause}
        ORDER BY distance {order_by}
        LIMIT {top_k}
        """
        rows = self._execute_sql(search_sql)

        collected: list[VectorSearchResult] = []
        for row in rows:
            chunk = Chunk.model_validate_json(row["chunk"])
            collected.append(
                VectorSearchResult(
                    score=row["distance"],
                    document_id=row["document_id"],
                    chunk=chunk,
                ),
            )
        return collected

    # ------------------------------------------------------------------
    # Document listing
    # ------------------------------------------------------------------

    async def list_documents(
        self,
        collection: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentSummary]:
        """List all distinct source documents indexed in a collection.

        Scans every row (vectors not needed) and aggregates by
        ``document_id``.  The first chunk encountered for each document
        supplies the ``source`` filename and the document-level
        ``metadata``.

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
        """
        where_clause = self._build_metadata_where(metadata_filter)
        list_sql = (
            f"SELECT document_id, chunk "
            f"FROM {self._qualified(collection)} {where_clause}"
        )
        rows = self._execute_sql(list_sql)

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
