from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from post_train_v2.src.artifacts.atomic import publish_json
from post_train_v2.src.artifacts.hashing import (
    canonical_json_bytes,
    sha256_canonical_json,
    sha256_config,
)

SCHEMA_VERSION = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "stage",
    "artifact_id",
    "created_at",
    "files",
    "parents",
    "config",
    "config_sha256",
    "model_path",
    "model_fingerprint",
    "global_seed",
    "seed_derivation_version",
    "git_revision",
    "runtime_versions",
    "stage_metadata",
}


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], structure_name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ValueError(f"{structure_name} keys mismatch ({'; '.join(details)})")


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain valid JSON values") from error


def _normalize_created_at(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat().replace("+00:00", "Z")
    if not isinstance(value, str):
        raise ValueError("created_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("artifact relative path must be a non-empty string")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("artifact relative path must not be absolute")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("artifact relative path must not traverse parents")
    if posix_path == PurePosixPath(".") or windows_path == PureWindowsPath("."):
        raise ValueError("artifact relative path must identify a file")
    if "\\" in value or value != posix_path.as_posix():
        raise ValueError(
            "artifact relative path must use canonical POSIX syntax"
        )
    return value


def _current_git_revision() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    sha256: str
    byte_size: int
    row_count: int
    field_schema: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_path", _validate_relative_path(self.relative_path)
        )
        object.__setattr__(
            self, "sha256", _require_sha256(self.sha256, "file SHA-256")
        )
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        if not isinstance(self.field_schema, Mapping):
            raise ValueError("field_schema must be an object")
        object.__setattr__(
            self,
            "field_schema",
            _json_copy(dict(self.field_schema), "field_schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "row_count": self.row_count,
            "field_schema": _json_copy(self.field_schema, "field_schema"),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArtifactFile:
        if not isinstance(value, Mapping):
            raise ValueError("artifact file must be an object")
        _require_exact_keys(
            value,
            {
                "relative_path",
                "sha256",
                "byte_size",
                "row_count",
                "field_schema",
            },
            "artifact file",
        )
        return cls(
            relative_path=value["relative_path"],
            sha256=value["sha256"],
            byte_size=value["byte_size"],
            row_count=value["row_count"],
            field_schema=value["field_schema"],
        )


@dataclass(frozen=True)
class ParentArtifact:
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _require_nonempty_string(self.artifact_id, "parent artifact_id"),
        )
        object.__setattr__(
            self, "sha256", _require_sha256(self.sha256, "parent SHA-256")
        )

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Any) -> ParentArtifact:
        if not isinstance(value, Mapping):
            raise ValueError("parent artifact must be an object")
        _require_exact_keys(value, {"artifact_id", "sha256"}, "parent artifact")
        return cls(artifact_id=value["artifact_id"], sha256=value["sha256"])


def _identity_payload(
    *,
    artifact_type: str,
    stage: str,
    files: Sequence[ArtifactFile],
    parents: Sequence[ParentArtifact],
    config: Mapping[str, Any],
    config_sha256: str,
    model_path: str | None,
    model_fingerprint: str | None,
    global_seed: int,
    seed_derivation_version: str,
    git_revision: str,
    runtime_versions: Mapping[str, str],
    stage_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "stage": stage,
        "files": [artifact_file.to_dict() for artifact_file in files],
        "parents": [parent.to_dict() for parent in parents],
        "config": _json_copy(dict(config), "config"),
        "config_sha256": config_sha256,
        "model_path": model_path,
        "model_fingerprint": model_fingerprint,
        "global_seed": global_seed,
        "seed_derivation_version": seed_derivation_version,
        "git_revision": git_revision,
        "runtime_versions": _json_copy(
            dict(runtime_versions), "runtime_versions"
        ),
        "stage_metadata": _json_copy(dict(stage_metadata), "stage_metadata"),
    }


@dataclass(frozen=True)
class ManifestV2:
    schema_version: int
    artifact_type: str
    stage: str
    artifact_id: str
    created_at: str
    files: tuple[ArtifactFile, ...]
    parents: tuple[ParentArtifact, ...]
    config: dict[str, Any]
    config_sha256: str
    model_path: str | None
    model_fingerprint: str | None
    global_seed: int
    seed_derivation_version: str
    git_revision: str
    runtime_versions: dict[str, str]
    stage_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("schema_version must be 2")
        object.__setattr__(
            self,
            "artifact_type",
            _require_nonempty_string(self.artifact_type, "artifact_type"),
        )
        object.__setattr__(
            self, "stage", _require_nonempty_string(self.stage, "stage")
        )
        object.__setattr__(
            self,
            "artifact_id",
            _require_sha256(self.artifact_id, "artifact_id"),
        )
        object.__setattr__(
            self, "created_at", _normalize_created_at(self.created_at)
        )
        if not isinstance(self.files, (list, tuple)):
            raise ValueError("files must be a list")
        if not isinstance(self.parents, (list, tuple)):
            raise ValueError("parents must be a list")
        files = tuple(self.files)
        parents = tuple(self.parents)
        if any(not isinstance(item, ArtifactFile) for item in files):
            raise ValueError("files must contain ArtifactFile values")
        if any(not isinstance(item, ParentArtifact) for item in parents):
            raise ValueError("parents must contain ParentArtifact values")
        file_paths = [item.relative_path for item in files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("duplicate file relative path")
        parent_ids = [item.artifact_id for item in parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("duplicate parent artifact_id")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "parents", parents)

        if not isinstance(self.config, Mapping):
            raise ValueError("config must be an object")
        config = _json_copy(dict(self.config), "config")
        object.__setattr__(self, "config", config)
        _require_sha256(self.config_sha256, "config hash")
        if self.config_sha256 != sha256_config(config):
            raise ValueError("stored config hash does not match config snapshot")

        if self.model_path is not None:
            _require_nonempty_string(self.model_path, "model_path")
        if self.model_fingerprint is not None:
            _require_sha256(self.model_fingerprint, "model fingerprint")
        if type(self.global_seed) is not int or self.global_seed < 0:
            raise ValueError("global_seed must be a non-negative integer")
        _require_nonempty_string(
            self.seed_derivation_version, "seed_derivation_version"
        )
        _require_nonempty_string(self.git_revision, "git_revision")

        if not isinstance(self.runtime_versions, Mapping):
            raise ValueError("runtime_versions must be an object")
        runtime_versions = dict(self.runtime_versions)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(version, str)
            or not version
            for key, version in runtime_versions.items()
        ):
            raise ValueError(
                "runtime_versions must map non-empty strings to strings"
            )
        object.__setattr__(
            self,
            "runtime_versions",
            _json_copy(runtime_versions, "runtime_versions"),
        )

        if not isinstance(self.stage_metadata, Mapping):
            raise ValueError("stage_metadata must be an object")
        object.__setattr__(
            self,
            "stage_metadata",
            _json_copy(dict(self.stage_metadata), "stage_metadata"),
        )

        expected_artifact_id = sha256_canonical_json(self.identity_dict())
        if self.artifact_id != expected_artifact_id:
            raise ValueError("artifact_id does not match manifest identity")

    @classmethod
    def build(
        cls,
        *,
        artifact_type: str,
        stage: str,
        files: Sequence[ArtifactFile],
        parents: Sequence[ParentArtifact],
        config: Mapping[str, Any],
        stage_metadata: Mapping[str, Any],
        model_path: str | None = None,
        model_fingerprint: str | None = None,
        global_seed: int | None = None,
        seed_derivation_version: str = "v1",
        git_revision: str | None = None,
        runtime_versions: Mapping[str, str] | None = None,
        created_at: datetime | str | None = None,
    ) -> ManifestV2:
        if not isinstance(config, Mapping):
            raise ValueError("config must be an object")
        config_copy = _json_copy(dict(config), "config")
        if global_seed is None:
            configured_seed = config_copy.get("seed", 0)
            global_seed = (
                configured_seed if type(configured_seed) is int else 0
            )
        files_tuple = tuple(files)
        parents_tuple = tuple(parents)
        runtime_copy = dict(runtime_versions or {})
        metadata_copy = (
            dict(stage_metadata)
            if isinstance(stage_metadata, Mapping)
            else stage_metadata
        )
        config_hash = sha256_config(config_copy)
        identity = _identity_payload(
            artifact_type=artifact_type,
            stage=stage,
            files=files_tuple,
            parents=parents_tuple,
            config=config_copy,
            config_sha256=config_hash,
            model_path=model_path,
            model_fingerprint=model_fingerprint,
            global_seed=global_seed,
            seed_derivation_version=seed_derivation_version,
            git_revision=git_revision or _current_git_revision(),
            runtime_versions=runtime_copy,
            stage_metadata=metadata_copy,
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_type=artifact_type,
            stage=stage,
            artifact_id=sha256_canonical_json(identity),
            created_at=_normalize_created_at(created_at),
            files=files_tuple,
            parents=parents_tuple,
            config=config_copy,
            config_sha256=config_hash,
            model_path=model_path,
            model_fingerprint=model_fingerprint,
            global_seed=global_seed,
            seed_derivation_version=seed_derivation_version,
            git_revision=identity["git_revision"],
            runtime_versions=runtime_copy,
            stage_metadata=metadata_copy,
        )

    @classmethod
    def parse(cls, value: Any) -> ManifestV2:
        if not isinstance(value, Mapping):
            raise ValueError("manifest must be an object")
        _require_exact_keys(value, MANIFEST_KEYS, "manifest")
        if type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("schema_version must be 2")
        if not isinstance(value["files"], list):
            raise ValueError("files must be a list")
        if not isinstance(value["parents"], list):
            raise ValueError("parents must be a list")
        if not isinstance(value["config"], Mapping):
            raise ValueError("config must be an object")
        if not isinstance(value["runtime_versions"], Mapping):
            raise ValueError("runtime_versions must be an object")
        if not isinstance(value["stage_metadata"], Mapping):
            raise ValueError("stage_metadata must be an object")
        return cls(
            schema_version=value["schema_version"],
            artifact_type=value["artifact_type"],
            stage=value["stage"],
            artifact_id=value["artifact_id"],
            created_at=value["created_at"],
            files=tuple(ArtifactFile.from_dict(item) for item in value["files"]),
            parents=tuple(
                ParentArtifact.from_dict(item) for item in value["parents"]
            ),
            config=dict(value["config"]),
            config_sha256=value["config_sha256"],
            model_path=value["model_path"],
            model_fingerprint=value["model_fingerprint"],
            global_seed=value["global_seed"],
            seed_derivation_version=value["seed_derivation_version"],
            git_revision=value["git_revision"],
            runtime_versions=dict(value["runtime_versions"]),
            stage_metadata=dict(value["stage_metadata"]),
        )

    @classmethod
    def from_dict(cls, value: Any) -> ManifestV2:
        return cls.parse(value)

    def identity_dict(self) -> dict[str, Any]:
        return _identity_payload(
            artifact_type=self.artifact_type,
            stage=self.stage,
            files=self.files,
            parents=self.parents,
            config=self.config,
            config_sha256=self.config_sha256,
            model_path=self.model_path,
            model_fingerprint=self.model_fingerprint,
            global_seed=self.global_seed,
            seed_derivation_version=self.seed_derivation_version,
            git_revision=self.git_revision,
            runtime_versions=self.runtime_versions,
            stage_metadata=self.stage_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "stage": self.stage,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "files": [artifact_file.to_dict() for artifact_file in self.files],
            "parents": [parent.to_dict() for parent in self.parents],
            "config": _json_copy(self.config, "config"),
            "config_sha256": self.config_sha256,
            "model_path": self.model_path,
            "model_fingerprint": self.model_fingerprint,
            "global_seed": self.global_seed,
            "seed_derivation_version": self.seed_derivation_version,
            "git_revision": self.git_revision,
            "runtime_versions": _json_copy(
                self.runtime_versions, "runtime_versions"
            ),
            "stage_metadata": _json_copy(
                self.stage_metadata, "stage_metadata"
            ),
        }

    def require_parent(self, artifact_id: str, sha256: str) -> ParentArtifact:
        for parent in self.parents:
            if parent.artifact_id == artifact_id:
                if parent.sha256 != sha256:
                    raise ValueError(
                        f"parent hash mismatch for artifact {artifact_id}"
                    )
                return parent
        raise ValueError(f"parent artifact not found: {artifact_id}")


def publish_manifest(path: str | Path, manifest: ManifestV2) -> None:
    if not isinstance(manifest, ManifestV2):
        raise TypeError("manifest must be a ManifestV2")
    validated = ManifestV2.parse(manifest.to_dict())
    publish_json(path, validated.to_dict())


def load_manifest(path: str | Path) -> ManifestV2:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest contains invalid JSON: {path}") from error
    return ManifestV2.parse(value)
