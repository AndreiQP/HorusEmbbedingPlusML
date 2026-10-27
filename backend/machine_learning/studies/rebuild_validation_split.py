"""Rebuild the LoveFraud02 portion of the external validation split.

The repository's original validation CSV lost a subset of the source speaker
markers.  This utility downloads the 83 authoritative LoveFraud02 files and
serializes every source message as either ``Innocent:`` (PFB*) or ``Suspect:``
(all other source speaker identifiers).

Only the 83 rows listed in ``data/validation_lovefraud_manifest.json`` are
replaced.  The 1,000 Ham rows and the original row order are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = HERE / "data" / "validation_lovefraud_manifest.json"
DEFAULT_INPUT = REPOSITORY_ROOT / "backend/datasets/dataset_validation/raw/all_data.csv"
DEFAULT_BACKUP = (
    REPOSITORY_ROOT
    / "backend/datasets/dataset_validation/raw/all_data.pre_speaker_reconstruction.csv"
)
PUBLIC_FILE_URL = (
    "https://data.mendeley.com/public-files/datasets/kmhvb4x5d8/files/"
    "{file_id}/file_downloaded"
)
SOURCE_MARKER = re.compile(
    r"^(?P<speaker>PFB\d*|[A-Za-z][A-Za-z_-]*\d{2})\s*\.?$",
    re.IGNORECASE,
)
CANONICAL_MARKER = re.compile(r"(?m)^(Innocent|Suspect): ")


@dataclass(frozen=True)
class SourceMessage:
    source_speaker: str
    role: str
    text: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/plain,*/*",
            "Referer": "https://data.mendeley.com/datasets/kmhvb4x5d8/1",
            "User-Agent": "Mozilla/5.0 HorusValidationRebuilder/1.0",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def ensure_sources(manifest: dict, source_dir: Path) -> None:
    for row in manifest["rows"]:
        destination = source_dir / row["source_file"]
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        _download(
            PUBLIC_FILE_URL.format(file_id=row["source_file_id"]),
            destination,
        )


def parse_source_file(path: Path) -> list[SourceMessage]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    messages: list[SourceMessage] = []
    speaker: str | None = None
    content: list[str] = []

    def flush() -> None:
        nonlocal content
        if speaker is None:
            content = []
            return
        text = " ".join(part.strip() for part in content if part.strip()).strip()
        if text:
            role = "Innocent" if speaker.upper().startswith("PFB") else "Suspect"
            messages.append(SourceMessage(speaker, role, text))
        content = []

    for raw_line in raw.replace("\r", "").split("\n"):
        line = raw_line.strip().lstrip("\ufeff")
        marker = SOURCE_MARKER.fullmatch(line)
        if marker:
            flush()
            speaker = marker.group("speaker")
        elif speaker is not None:
            content.append(raw_line)
    flush()

    if not messages:
        raise ValueError(f"Nenhuma mensagem reconhecida em {path}")
    if not any(message.role == "Suspect" for message in messages):
        raise ValueError(f"Nenhuma mensagem Suspect reconhecida em {path}")
    if not any(message.role == "Innocent" for message in messages):
        raise ValueError(f"Nenhuma mensagem Innocent reconhecida em {path}")
    return messages


def serialize_messages(messages: list[SourceMessage]) -> str:
    return "\n".join(f"{message.role}: {message.text}" for message in messages)


def validate_rebuilt_frame(frame: pd.DataFrame, manifest: dict) -> dict:
    if list(frame.columns) != ["text", "label"]:
        raise ValueError(f"Colunas inesperadas: {list(frame.columns)}")
    if len(frame) != 1083:
        raise ValueError(f"Esperadas 1083 conversas; encontradas {len(frame)}")
    if frame["label"].value_counts().to_dict() != {0: 1000, 1: 83}:
        raise ValueError(f"Distribuição de rótulos inválida: {frame['label'].value_counts().to_dict()}")

    rows = [int(item["row_index"]) for item in manifest["rows"]]
    if len(rows) != 83 or len(set(rows)) != 83:
        raise ValueError("O manifesto não contém exatamente 83 índices únicos")
    if set(rows) != set(frame.index[frame["label"] == 1].tolist()):
        raise ValueError("Os índices Scam do CSV divergem do manifesto")

    suspect_messages = 0
    innocent_messages = 0
    for row_index in rows:
        text = str(frame.at[row_index, "text"])
        markers = CANONICAL_MARKER.findall(text)
        suspect_messages += markers.count("Suspect")
        innocent_messages += markers.count("Innocent")
        invalid_lines = [
            line for line in text.splitlines()
            if line and not re.match(r"^(?:Innocent|Suspect): ", line)
        ]
        if invalid_lines:
            raise ValueError(
                f"Linha canônica inválida na conversa {row_index}: {invalid_lines[0]!r}"
            )

    return {
        "rows": len(frame),
        "ham_conversations": int((frame["label"] == 0).sum()),
        "scam_conversations": int((frame["label"] == 1).sum()),
        "suspect_messages": suspect_messages,
        "innocent_messages": innocent_messages,
    }


def rebuild(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("expected_conversations") != 83:
        raise ValueError("Manifesto LoveFraud02 inválido")
    ensure_sources(manifest, source_dir)

    frame = pd.read_csv(input_path)
    original_ham_hashes = {
        int(index): _sha256_text(str(text))
        for index, text in frame.loc[frame["label"] == 0, "text"].items()
    }

    for item in manifest["rows"]:
        row_index = int(item["row_index"])
        if int(frame.at[row_index, "label"]) != 1:
            raise ValueError(f"A linha {row_index} não é Scam")
        source_path = source_dir / item["source_file"]
        frame.at[row_index, "text"] = serialize_messages(parse_source_file(source_path))

    for index, expected_hash in original_ham_hashes.items():
        actual_hash = _sha256_text(str(frame.at[index, "text"]))
        if actual_hash != expected_hash:
            raise ValueError(f"A conversa Ham {index} foi alterada")

    summary = validate_rebuilt_frame(frame, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="validation_rebuilt_",
        dir=output_path.parent,
        encoding="utf-8",
        newline="",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        frame.to_csv(temporary, index=False)
    temporary_path.replace(output_path)
    return frame, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        parser.error(f"Arquivo de entrada inexistente: {input_path}")

    if output_path == input_path and not args.no_backup:
        backup_path = args.backup.resolve()
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"[backup] {backup_path}")
        else:
            print(f"[backup] preservado: {backup_path}")

    if args.source_dir is None:
        source_dir = Path(tempfile.gettempdir()) / "lovefraud02_v1"
    else:
        source_dir = args.source_dir.resolve()

    _, summary = rebuild(
        input_path=input_path,
        output_path=output_path,
        manifest_path=args.manifest.resolve(),
        source_dir=source_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[output] {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
