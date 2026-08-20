import pytest

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.Database.sql_models import Frame, OCR, ObjectDetection


class EmptyScalarResult:
    def all(self):
        return []


class QueryCapturingSession:
    def __init__(self) -> None:
        self.queried_entities = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def get(self, model, entity_id):
        return object()

    def scalars(self, statement):
        self.queried_entities.append(statement.column_descriptions[0]["entity"])
        return EmptyScalarResult()


@pytest.mark.parametrize(
    ("modality", "expected_entity"),
    [("ocr", OCR), ("object", ObjectDetection)],
)
def test_modality_query_does_not_materialize_standalone_frames(
    modality,
    expected_entity,
) -> None:
    session = QueryCapturingSession()
    manager = PostgreManager.__new__(PostgreManager)
    manager.session_factory = lambda: session

    manager.get_evidence_by_video_id_and_time(
        "video-1",
        1000,
        2000,
        modalities={modality},
        limits={modality: 5},
    )

    assert session.queried_entities == [expected_entity]
    assert Frame not in session.queried_entities
