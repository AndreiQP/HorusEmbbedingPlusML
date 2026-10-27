from pathlib import Path

from machine_learning.studies.rebuild_validation_split import (
    parse_source_file,
    serialize_messages,
)
from machine_learning.studies.rebuild_validation_bge_messages import split_messages
from datasets.dataset_train.process import process_text


def test_source_markers_are_converted_without_losing_consecutive_messages(tmp_path: Path):
    source = tmp_path / "OliverAntero.txt"
    source.write_text(
        "\n".join(
            [
                "PFB",
                "Happy Sunday, Oliver.",
                "",
                "Eng13",
                "My name is Oliver Antero.",
                "",
                "Eng13.",
                "I work offshore.",
                "",
                "PFB02",
                "Where are you from?",
            ]
        ),
        encoding="utf-8",
    )

    messages = parse_source_file(source)

    assert [message.role for message in messages] == [
        "Innocent",
        "Suspect",
        "Suspect",
        "Innocent",
    ]
    assert serialize_messages(messages) == "\n".join(
        [
            "Innocent: Happy Sunday, Oliver.",
            "Suspect: My name is Oliver Antero.",
            "Suspect: I work offshore.",
            "Innocent: Where are you from?",
        ]
    )


def test_suspect_projection_keeps_consecutive_suspect_messages():
    row = {
        "dataset": "external_validation",
        "text": "\n".join(
            [
                "Innocent: Hello.",
                "Suspect: First scammer message.",
                "Suspect: Second scammer message.",
                "Innocent: Goodbye.",
            ]
        ),
    }

    assert process_text(row) == "\n".join(
        [
            "Suspect: First scammer message.",
            "Suspect: Second scammer message.",
        ]
    )


def test_bge_message_split_preserves_prefix_and_consecutive_speaker():
    conversation = "\n".join(
        [
            "Innocent: First message.",
            "Suspect: Second message.",
            "Suspect: Third message.",
        ]
    )

    assert split_messages(conversation) == [
        "Innocent: First message.",
        "Suspect: Second message.",
        "Suspect: Third message.",
    ]
