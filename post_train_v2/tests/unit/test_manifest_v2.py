from copy import deepcopy
from datetime import datetime, timezone

import pytest

from post_train_v2.src.artifacts import ManifestV2 as PublicManifestV2
from post_train_v2.src.artifacts.hashing import sha256_config
from post_train_v2.src.artifacts.manifest import (
    ArtifactFile,
    ManifestV2,
    ParentArtifact,
    load_manifest,
    publish_manifest,
)


FILE_HASH = "a" * 64
PARENT_HASH = "d" * 64
MODEL_HASH = "e" * 64


def build_manifest(**overrides):
    arguments = {
        "artifact_type": "dataset",
        "stage": "build_source",
        "files": [
            ArtifactFile(
                "source.jsonl",
                FILE_HASH,
                4,
                1,
                {"id": "string"},
            )
        ],
        "parents": [ParentArtifact("raw-train", PARENT_HASH)],
        "config": {"seed": 42, "name": "雪"},
        "stage_metadata": {"num_source": 1},
        "model_path": "models/teacher",
        "model_fingerprint": MODEL_HASH,
        "global_seed": 42,
        "seed_derivation_version": "v1",
        "git_revision": "723bb48",
        "runtime_versions": {"python": "3.11.15", "pytest": "8.3.5"},
    }
    arguments.update(overrides)
    return ManifestV2.build(**arguments)


def test_manifest_round_trip_rejects_changed_parent_hash(tmp_path):
    manifest = ManifestV2.build(
        artifact_type="dataset",
        stage="build_source",
        files=[ArtifactFile("source.jsonl", "a" * 64, 4, 1, {"id": "string"})],
        parents=[ParentArtifact("raw-train", "d" * 64)],
        config={"seed": 42},
        stage_metadata={"num_source": 1},
    )
    path = tmp_path / "manifest.json"
    publish_manifest(path, manifest)
    loaded = load_manifest(path)
    with pytest.raises(ValueError, match="parent hash"):
        loaded.require_parent("raw-train", "b" * 64)


def test_manifest_round_trip_preserves_complete_contract(tmp_path):
    manifest = build_manifest()
    path = tmp_path / "nested" / "manifest.json"

    publish_manifest(path, manifest)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded.to_dict() == {
        "schema_version": 2,
        "artifact_type": "dataset",
        "stage": "build_source",
        "artifact_id": manifest.artifact_id,
        "created_at": manifest.created_at,
        "files": [
            {
                "relative_path": "source.jsonl",
                "sha256": FILE_HASH,
                "byte_size": 4,
                "row_count": 1,
                "field_schema": {"id": "string"},
            }
        ],
        "parents": [{"artifact_id": "raw-train", "sha256": PARENT_HASH}],
        "config": {"seed": 42, "name": "雪"},
        "config_sha256": sha256_config({"seed": 42, "name": "雪"}),
        "model_path": "models/teacher",
        "model_fingerprint": MODEL_HASH,
        "global_seed": 42,
        "seed_derivation_version": "v1",
        "git_revision": "723bb48",
        "runtime_versions": {"python": "3.11.15", "pytest": "8.3.5"},
        "stage_metadata": {"num_source": 1},
    }
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("preexisting", (False, True))
def test_publish_manifest_revalidates_mutated_nested_state_before_write(
    tmp_path, preexisting
):
    path = tmp_path / "manifest.json"
    original = b'{"existing":true}'
    if preexisting:
        path.write_bytes(original)
    manifest = build_manifest(config={"nested": {"seed": 42}})
    manifest.config["nested"]["seed"] = 7

    with pytest.raises(ValueError, match="config hash"):
        publish_manifest(path, manifest)

    if preexisting:
        assert path.read_bytes() == original
    else:
        assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_artifact_id_is_stable_across_creation_times():
    first = build_manifest(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    second = build_manifest(
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc)
    )

    assert first.created_at != second.created_at
    assert first.artifact_id == second.artifact_id


def test_manifest_parse_rejects_changed_artifact_identity():
    data = build_manifest().to_dict()
    data["stage_metadata"]["num_source"] = 2

    with pytest.raises(ValueError, match="artifact_id"):
        ManifestV2.parse(data)


def test_manifest_parse_is_public_and_preserves_contract():
    manifest = build_manifest()

    assert PublicManifestV2.parse(manifest.to_dict()) == manifest


def test_require_parent_rejects_missing_parent():
    manifest = build_manifest()

    with pytest.raises(ValueError, match="parent artifact"):
        manifest.require_parent("missing-parent", PARENT_HASH)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 1),
        ("files", "not-a-list"),
        ("parents", "not-a-list"),
        ("config", []),
        ("stage_metadata", []),
        ("runtime_versions", []),
    ),
)
def test_manifest_parse_rejects_malformed_root_structures(field, value):
    data = build_manifest().to_dict()
    data[field] = value

    with pytest.raises(ValueError):
        ManifestV2.from_dict(data)


def test_manifest_parse_rejects_mismatched_config_hash():
    data = build_manifest().to_dict()
    data["config_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="config hash"):
        ManifestV2.from_dict(data)


def test_manifest_rejects_duplicate_parent_ids():
    parent = ParentArtifact("raw-train", PARENT_HASH)

    with pytest.raises(ValueError, match="duplicate parent"):
        build_manifest(parents=[parent, parent])


def test_manifest_rejects_duplicate_file_paths():
    artifact_file = ArtifactFile(
        "source.jsonl", FILE_HASH, 4, 1, {"id": "string"}
    )

    with pytest.raises(ValueError, match="duplicate file"):
        build_manifest(files=[artifact_file, artifact_file])


def test_manifest_rejects_windows_case_insensitive_duplicate_file_paths():
    with pytest.raises(ValueError, match="duplicate file"):
        build_manifest(
            files=[
                ArtifactFile(
                    "nested/source.jsonl",
                    FILE_HASH,
                    4,
                    1,
                    {"id": "string"},
                ),
                ArtifactFile(
                    "nested/SOURCE.jsonl",
                    "b" * 64,
                    4,
                    1,
                    {"id": "string"},
                ),
            ]
        )


@pytest.mark.parametrize("sha256", ("abc", "g" * 64, "A" * 64, 123))
def test_artifact_file_rejects_invalid_sha256(sha256):
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactFile("source.jsonl", sha256, 4, 1, {"id": "string"})


@pytest.mark.parametrize("sha256", ("abc", "g" * 64, "A" * 64, 123))
def test_parent_artifact_rejects_invalid_sha256(sha256):
    with pytest.raises(ValueError, match="SHA-256"):
        ParentArtifact("raw-train", sha256)


@pytest.mark.parametrize(
    "relative_path",
    (
        "/absolute/source.jsonl",
        "C:\\absolute\\source.jsonl",
        "C:relative.jsonl",
        "\\rooted.jsonl",
        "../source.jsonl",
        "nested/../../source.jsonl",
        "",
    ),
)
def test_artifact_file_rejects_unsafe_relative_paths(relative_path):
    with pytest.raises(ValueError, match="relative path"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


@pytest.mark.parametrize(
    "relative_path",
    (
        "nested//source.jsonl",
        "nested/./source.jsonl",
        "nested\\source.jsonl",
    ),
)
def test_artifact_file_rejects_noncanonical_posix_paths(relative_path):
    with pytest.raises(ValueError, match="canonical POSIX"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


@pytest.mark.parametrize(
    "relative_path",
    (
        "CON",
        "con.jsonl",
        "nested/AUX",
        "nested/prn.txt",
        "NUL.jsonl",
        "COM1",
        "nested/com9.bin",
        "LPT1.txt",
        "nested/lpt9",
        "COM¹.txt",
        "nested/lpt³.log",
    ),
)
def test_artifact_file_rejects_windows_reserved_device_names(relative_path):
    with pytest.raises(ValueError, match="reserved Windows device"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


@pytest.mark.parametrize(
    "relative_path",
    (
        "source.",
        "source ",
        "nested/name. /source.jsonl",
        "nested/name /source.jsonl",
    ),
)
def test_artifact_file_rejects_windows_trailing_dot_or_space(relative_path):
    with pytest.raises(ValueError, match="trailing dot or space"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


@pytest.mark.parametrize(
    "relative_path",
    (
        "nested/a<b.jsonl",
        "nested/a>b.jsonl",
        'nested/a"b.jsonl',
        "nested/a:b.jsonl",
        "nested/a|b.jsonl",
        "nested/a?b.jsonl",
        "nested/a*b.jsonl",
        "nested/control\x1f.jsonl",
    ),
)
def test_artifact_file_rejects_windows_invalid_component_characters(
    relative_path,
):
    with pytest.raises(ValueError, match="Windows-safe"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


def test_artifact_file_rejects_windows_component_over_255_utf16_units():
    relative_path = f"nested/{'a' * 256}.jsonl"

    with pytest.raises(ValueError, match="Windows-safe"):
        ArtifactFile(relative_path, FILE_HASH, 4, 1, {"id": "string"})


@pytest.mark.parametrize(
    "alias",
    (
        "nested//source.jsonl",
        "nested/./source.jsonl",
        "nested\\source.jsonl",
    ),
)
def test_manifest_parse_rejects_duplicate_path_aliases(alias):
    data = build_manifest(
        files=[
            ArtifactFile(
                "nested/source.jsonl",
                FILE_HASH,
                4,
                1,
                {"id": "string"},
            )
        ]
    ).to_dict()
    aliased_file = deepcopy(data["files"][0])
    aliased_file["relative_path"] = alias
    data["files"].append(aliased_file)

    with pytest.raises(ValueError, match="canonical POSIX"):
        ManifestV2.parse(data)


@pytest.mark.parametrize(
    ("byte_size", "row_count"),
    ((-1, 0), (0, -1), (True, 0), (0, False)),
)
def test_artifact_file_rejects_invalid_sizes_and_counts(byte_size, row_count):
    with pytest.raises(ValueError):
        ArtifactFile(
            "source.jsonl",
            FILE_HASH,
            byte_size,
            row_count,
            {"id": "string"},
        )


def test_manifest_parse_rejects_malformed_nested_structures():
    data = build_manifest().to_dict()
    data["files"][0]["field_schema"] = []

    with pytest.raises(ValueError, match="field_schema"):
        ManifestV2.from_dict(data)


def test_manifest_copies_mutable_inputs():
    config = {"nested": {"seed": 42}}
    metadata = {"shards": [{"index": 0}]}
    runtime_versions = {"python": "3.11.15"}
    manifest = build_manifest(
        config=config,
        stage_metadata=metadata,
        runtime_versions=runtime_versions,
    )
    before = deepcopy(manifest.to_dict())

    config["nested"]["seed"] = 7
    metadata["shards"][0]["index"] = 1
    runtime_versions["python"] = "changed"

    assert manifest.to_dict() == before
