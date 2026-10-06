import json
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import Json
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from pydantic import BaseModel
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionData import PredictionData
from trainable_entity_extractor.domain.Suggestion import Suggestion

from config import POSTGRES_DSN, MATERIALS_HOURS_TO_KEEP
from domain.ParagraphExtractionData import ParagraphExtractionData
from ports.PersistenceRepository import PersistenceRepository

DDL = """
CREATE TABLE IF NOT EXISTS labeled_data (
    run_name TEXT NOT NULL,
    extraction_name TEXT NOT NULL,
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS labeled_data_identity_idx ON labeled_data (run_name, extraction_name);
CREATE TABLE IF NOT EXISTS prediction_data (
    run_name TEXT NOT NULL,
    extraction_name TEXT NOT NULL,
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS prediction_data_identity_idx ON prediction_data (run_name, extraction_name);
CREATE TABLE IF NOT EXISTS paragraphs_from_languages (
    run_name TEXT NOT NULL,
    extraction_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS paragraphs_from_languages_identity_idx ON paragraphs_from_languages (run_name, extraction_name);
ALTER TABLE paragraphs_from_languages ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE TABLE IF NOT EXISTS suggestions (
    run_name TEXT NOT NULL,
    extraction_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS suggestions_identity_created_idx ON suggestions (run_name, extraction_name, created_at);
CREATE TABLE IF NOT EXISTS paragraph_extraction_data (
    run_name TEXT NOT NULL,
    extraction_name TEXT NOT NULL,
    data JSONB NOT NULL,
    CONSTRAINT paragraph_extraction_data_identity_key UNIQUE (run_name, extraction_name)
);
"""


class PostgresPersistenceRepository(PersistenceRepository):

    _tables_ensured = False

    def __init__(self):
        self.conn = None
        if not PostgresPersistenceRepository._tables_ensured:
            self._ensure_tables()
            PostgresPersistenceRepository._tables_ensured = True

    def _connect(self):
        conn = psycopg2.connect(POSTGRES_DSN)
        conn.autocommit = True
        return conn

    def _ensure_tables(self):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(DDL)
        finally:
            conn.close()

    def _get_connection(self):
        if self.conn is None or self.conn.closed:
            self.conn = self._connect()
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @staticmethod
    def inject_extractor_identifier(extraction_identifier: ExtractionIdentifier, data: dict):
        data["run_name"] = extraction_identifier.run_name
        data["extraction_name"] = extraction_identifier.extraction_name
        return data

    def save_data(self, extraction_identifier: ExtractionIdentifier, data: BaseModel, table_name: str):
        data_dict = data.model_dump()
        data_dict = self.inject_extractor_identifier(extraction_identifier, data_dict)
        self._insert_data(extraction_identifier, data_dict, table_name)

    def _insert_data(
        self, extraction_identifier: ExtractionIdentifier, data_dict: dict, table_name: str, upsert: bool = False
    ):
        query = "INSERT INTO {table} (run_name, extraction_name, data) VALUES (%s, %s, %s)".format(table=table_name)
        if upsert:
            query += " ON CONFLICT (run_name, extraction_name) DO UPDATE SET data = EXCLUDED.data"

        with self._get_connection().cursor() as cur:
            cur.execute(query, (extraction_identifier.run_name, extraction_identifier.extraction_name, Json(data_dict)))

    def save_prediction_data(self, extraction_identifier: ExtractionIdentifier, prediction_data: PredictionData):
        self.save_data(extraction_identifier, prediction_data, "prediction_data")

    def _load_all(self, table_name: str, extraction_identifier: ExtractionIdentifier, model: type[BaseModel]):
        with self._get_connection().cursor() as cur:
            cur.execute(
                f"DELETE FROM {table_name} WHERE run_name = %s AND extraction_name = %s RETURNING data",
                (extraction_identifier.run_name, extraction_identifier.extraction_name),
            )
            documents = [row[0] for row in cur.fetchall()]

        return [model(**document) for document in documents]

    def load_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionData]:
        return self._load_all("prediction_data", extraction_identifier, PredictionData)

    def load_and_delete_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionData]:
        return self._load_all("prediction_data", extraction_identifier, PredictionData)

    def save_labeled_data(self, extraction_identifier: ExtractionIdentifier, labeled_data: LabeledData):
        self.save_data(extraction_identifier, labeled_data, "labeled_data")

    def delete_labeled_data(self, extraction_identifier: ExtractionIdentifier):
        with self._get_connection().cursor() as cur:
            cur.execute(
                "DELETE FROM labeled_data WHERE run_name = %s AND extraction_name = %s",
                (extraction_identifier.run_name, extraction_identifier.extraction_name),
            )

    def load_labeled_data(self, extraction_identifier: ExtractionIdentifier) -> list[LabeledData]:
        return self._load_all("labeled_data", extraction_identifier, LabeledData)

    def load_and_delete_labeled_data(self, extraction_identifier: ExtractionIdentifier) -> list[LabeledData]:
        return self._load_all("labeled_data", extraction_identifier, LabeledData)

    def save_suggestions(self, extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]):
        self.delete_expired_suggestions(extraction_identifier)

        created_at = datetime.now(timezone.utc)
        for suggestion in suggestions:
            data_dict = suggestion.model_dump()
            data_dict = self.inject_extractor_identifier(extraction_identifier, data_dict)
            data_dict["created_at"] = created_at.isoformat()

            with self._get_connection().cursor() as cur:
                cur.execute(
                    "INSERT INTO suggestions (run_name, extraction_name, created_at, data) VALUES (%s, %s, %s, %s)",
                    (extraction_identifier.run_name, extraction_identifier.extraction_name, created_at, Json(data_dict)),
                )

    def delete_expired_suggestions(self, extraction_identifier: ExtractionIdentifier):
        with self._get_connection().cursor() as cur:
            cur.execute(
                "DELETE FROM suggestions WHERE run_name = %s AND extraction_name = %s"
                " AND created_at < now() - make_interval(hours => %s)",
                (extraction_identifier.run_name, extraction_identifier.extraction_name, MATERIALS_HOURS_TO_KEEP),
            )

    def load_suggestions(self, extraction_identifier: ExtractionIdentifier) -> list[Suggestion]:
        self.delete_expired_suggestions(extraction_identifier)

        with self._get_connection().cursor() as cur:
            cur.execute(
                "SELECT data FROM suggestions WHERE run_name = %s AND extraction_name = %s"
                " AND created_at >= now() - make_interval(hours => %s) ORDER BY created_at",
                (extraction_identifier.run_name, extraction_identifier.extraction_name, MATERIALS_HOURS_TO_KEEP),
            )
            documents = [row[0] for row in cur.fetchall()]

        return [Suggestion(**document) for document in documents]

    def save_paragraph_extraction_data(
        self, extraction_identifier: ExtractionIdentifier, paragraph_extraction_data: ParagraphExtractionData
    ):
        data_dict = paragraph_extraction_data.model_dump()
        data_dict = self.inject_extractor_identifier(extraction_identifier, data_dict)
        self._insert_data(extraction_identifier, data_dict, "paragraph_extraction_data", upsert=True)

    def load_paragraph_extraction_data(
        self, extraction_identifier: ExtractionIdentifier
    ) -> Optional[ParagraphExtractionData]:
        with self._get_connection().cursor() as cur:
            cur.execute(
                "SELECT data FROM paragraph_extraction_data WHERE run_name = %s AND extraction_name = %s LIMIT 1",
                (extraction_identifier.run_name, extraction_identifier.extraction_name),
            )
            row = cur.fetchone()

        if row is None:
            return None
        return ParagraphExtractionData(**row[0])

    def save_paragraphs_from_language(
        self, extraction_identifier: ExtractionIdentifier, paragraphs_from_languages: ParagraphsFromLanguage
    ):
        created_at = datetime.now(timezone.utc)
        data_dict = paragraphs_from_languages.model_dump()
        data_dict = self.inject_extractor_identifier(extraction_identifier, data_dict)
        data_dict["created_at"] = created_at.isoformat()

        with self._get_connection().cursor() as cur:
            cur.execute(
                "INSERT INTO paragraphs_from_languages (run_name, extraction_name, created_at, data) "
                "VALUES (%s, %s, %s, %s)",
                (extraction_identifier.run_name, extraction_identifier.extraction_name, created_at, Json(data_dict)),
            )

    def delete_expired_paragraphs(self, extraction_identifier: ExtractionIdentifier):
        with self._get_connection().cursor() as cur:
            cur.execute(
                "DELETE FROM paragraphs_from_languages WHERE run_name = %s AND extraction_name = %s"
                " AND created_at < now() - make_interval(hours => %s)",
                (extraction_identifier.run_name, extraction_identifier.extraction_name, MATERIALS_HOURS_TO_KEEP),
            )

    def load_paragraphs_from_languages(self, extraction_identifier: ExtractionIdentifier) -> list[ParagraphsFromLanguage]:
        self.delete_expired_paragraphs(extraction_identifier)

        with self._get_connection().cursor() as cur:
            cur.execute(
                "SELECT data FROM paragraphs_from_languages WHERE run_name = %s AND extraction_name = %s"
                " AND created_at >= now() - make_interval(hours => %s) ORDER BY created_at",
                (extraction_identifier.run_name, extraction_identifier.extraction_name, MATERIALS_HOURS_TO_KEEP),
            )
            documents = [row[0] for row in cur.fetchall()]

        return [ParagraphsFromLanguage(**document) for document in documents]

    def delete_paragraphs_from_languages(self, extraction_identifier: ExtractionIdentifier):
        with self._get_connection().cursor() as cur:
            cur.execute(
                "DELETE FROM paragraphs_from_languages WHERE run_name = %s AND extraction_name = %s",
                (extraction_identifier.run_name, extraction_identifier.extraction_name),
            )

    def delete_prediction_data(self, extraction_identifier: ExtractionIdentifier, filters: list[dict[str, str]]):
        for one_filter in filters:
            with self._get_connection().cursor() as cur:
                cur.execute(
                    "DELETE FROM suggestions WHERE run_name = %s AND extraction_name = %s AND data @> %s::jsonb",
                    (
                        extraction_identifier.run_name,
                        extraction_identifier.extraction_name,
                        json.dumps(one_filter),
                    ),
                )
