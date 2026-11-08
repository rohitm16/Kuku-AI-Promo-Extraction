"""Safe screenplay file ingestion, validation, and extraction management for Streamlit."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st

from kuku_promo_system.config import DEFAULT_DATA_DIR
from kuku_promo_system.logger import logger

MAX_SINGLE_FILE_BYTES = 5 * 1024 * 1024  # 5 MB max per text file
MAX_ZIP_EXTRACTED_BYTES = 50 * 1024 * 1024  # 50 MB total extraction limit
MAX_FILES_IN_ZIP = 200


class ScreenplayFileManager:
    """Manages safe ingestion of screenplay files from user uploads or built-in archives."""

    @staticmethod
    def get_builtin_dataset_files() -> List[Path]:
        """Retrieve the canonical 50 screenplay files from the local project repository."""
        if not DEFAULT_DATA_DIR.exists():
            return []
        files = sorted(list(DEFAULT_DATA_DIR.glob("episode-*.txt")))
        return files

    @staticmethod
    def validate_screenplay_text(content: str, filename: str) -> Tuple[bool, Optional[str]]:
        """Validate that a file conforms to expected multi-track screenplay format."""
        if not content.strip():
            return False, f"File '{filename}' is empty."

        # Check for minimum expected screenplay markers (timestamps or scene headers)
        has_timestamps = "Start:" in content or "End:" in content or "(" in content and ":" in content
        has_dialogue_or_scene = "Scene" in content or ":" in content

        if not (has_timestamps and has_dialogue_or_scene):
            return (
                False,
                f"File '{filename}' lacks recognized screenplay markers (expected timestamps and dialogue/scene headers).",
            )

        return True, None

    @staticmethod
    def extract_zip_safely(zip_bytes: bytes, dest_dir: Path) -> Tuple[List[Path], List[str]]:
        """Safely extract screenplay text files from a ZIP archive with Zip-Slip path traversal prevention."""
        extracted_files: List[Path] = []
        errors: List[str] = []
        total_extracted_bytes = 0

        dest_dir_resolved = dest_dir.resolve()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_zip.write(zip_bytes)
            tmp_zip_path = Path(tmp_zip.name)

        try:
            with zipfile.ZipFile(tmp_zip_path, "r") as zf:
                infolist = zf.infolist()
                if len(infolist) > MAX_FILES_IN_ZIP:
                    return [], [f"ZIP contains {len(infolist)} files (maximum allowed is {MAX_FILES_IN_ZIP})."]

                for member in infolist:
                    # Skip directories
                    if member.is_dir():
                        continue

                    # Only accept .txt files
                    filename = Path(member.filename).name
                    if not filename.lower().endswith(".txt") or filename.startswith("."):
                        continue

                    # SECURITY: Prevent Zip-Slip / Path Traversal
                    if ".." in member.filename or member.filename.startswith(("/", "\\")):
                        errors.append(f"Security Alert: Blocked suspicious path traversal entry '{member.filename}'.")
                        continue

                    target_path = (dest_dir / filename).resolve()
                    if not str(target_path).startswith(str(dest_dir_resolved)):
                        errors.append(f"Security Alert: Blocked path escaping root '{member.filename}'.")
                        continue

                    # Check decompressed size
                    total_extracted_bytes += member.file_size
                    if total_extracted_bytes > MAX_ZIP_EXTRACTED_BYTES:
                        errors.append("Decompressed ZIP size exceeds safe limit (50 MB).")
                        break

                    # Extract file safely
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

                    extracted_files.append(target_path)

        except zipfile.BadZipFile:
            errors.append("Uploaded file is not a valid ZIP archive.")
        finally:
            if tmp_zip_path.exists():
                tmp_zip_path.unlink()

        return sorted(extracted_files), errors

    @staticmethod
    def save_uploaded_txt_files(uploaded_files: List[Any], dest_dir: Path) -> Tuple[List[Path], List[str]]:
        """Save and validate multiple user-uploaded .txt files."""
        saved_files: List[Path] = []
        errors: List[str] = []

        for uf in uploaded_files:
            if not uf.name.lower().endswith(".txt"):
                errors.append(f"Skipped '{uf.name}': only .txt screenplay files are supported.")
                continue

            content_bytes = uf.getvalue()
            if len(content_bytes) > MAX_SINGLE_FILE_BYTES:
                errors.append(f"Skipped '{uf.name}': file exceeds maximum size (5 MB).")
                continue

            try:
                content_str = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content_str = content_bytes.decode("latin-1")
                except Exception:
                    errors.append(f"Skipped '{uf.name}': unable to decode text encoding.")
                    continue

            is_valid, val_err = ScreenplayFileManager.validate_screenplay_text(content_str, uf.name)
            if not is_valid:
                errors.append(f"Warning: {val_err}")

            target_path = dest_dir / uf.name
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content_str)
            saved_files.append(target_path)

        return sorted(saved_files), errors
