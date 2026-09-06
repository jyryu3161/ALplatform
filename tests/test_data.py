"""Chemical identity, provenance and holdout isolation regression tests."""

import numpy as np
import pandas as pd
import pytest

from active_learning_docking.data import (
    LibraryValidationError,
    featurize,
    load_library,
    scaffold_keys,
    split_holdout,
)
from active_learning_docking.acquisition import select_indices


def write_library(tmp_path, rows, name="library.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["Catalog ID", "smiles", "docking_score"]).to_csv(path, index=False)
    return path


def test_identifiers_canonical_structures_and_provenance(tmp_path):
    path = write_library(tmp_path, [("009", "OCC", -5), ("NA", "CCN", -4), ("002", "CC", -3)])
    library, rejected = load_library(path)
    assert library["compound_id"].tolist() == ["002", "009", "NA"]
    assert library["smiles"].tolist() == ["CC", "CCO", "CCN"]
    assert library["source_row"].tolist() == [4, 2, 3]
    assert set(library["source_file"]) == {str(path.resolve())}
    assert not len(rejected)


def test_sorted_directory_and_duplicate_report(tmp_path):
    write_library(tmp_path, [("b", "OCC", -5)], "z.csv")
    write_library(tmp_path, [("a", "CCO", -5), ("c", "CC", -3)], "a.csv")
    library, rejected = load_library(tmp_path)
    assert library["compound_id"].tolist() == ["a", "c"]
    assert rejected["reason"].tolist() == ["duplicate_structure"]
    assert rejected["kept_compound_id"].tolist() == ["a"]
    assert rejected["source_file"].iloc[0].endswith("z.csv")


@pytest.mark.parametrize("rows,reason", [
    ([("a", "CCO", -5), ("a", "CCN", -5)], "conflicting_compound_id"),
    ([("a", "CCO", -5), ("a", "OCC", -6)], "conflicting_compound_id"),
    ([("a", "CCO", -5), ("b", "OCC", -6)], "conflicting_structure_score"),
])
@pytest.mark.parametrize("deduplicate", [True, False])
def test_conflicts_are_fatal_even_under_skip(tmp_path, rows, reason, deduplicate):
    path = write_library(tmp_path, rows)
    with pytest.raises(LibraryValidationError, match=reason) as error:
        load_library(path, invalid_policy="skip", deduplicate=deduplicate)
    assert error.value.rejections["reason"].iloc[-1] == reason


def test_invalid_rows_reported_and_not_all_zero_features(tmp_path):
    path = write_library(tmp_path, [
        ("a", "not-a-smiles", -5), ("b", "", -4), ("", "CC", -3),
        ("c", "CCN", "Infinity"), ("d", "CCC", "oops"), ("e", "CO", -2),
    ])
    with pytest.raises(LibraryValidationError, match="invalid_smiles"):
        load_library(path)
    library, rejected = load_library(path, invalid_policy="skip")
    assert library["compound_id"].tolist() == ["e"]
    assert set(rejected["reason"]) == {
        "invalid_smiles", "missing_smiles", "missing_compound_id", "nonfinite_score", "invalid_score",
    }
    assert rejected["source_row"].tolist() == [2, 3, 4, 5, 6]


def test_all_invalid_is_failure_with_report(tmp_path):
    path = write_library(tmp_path, [("a", "invalid", -1)])
    with pytest.raises(LibraryValidationError, match="No valid compounds") as error:
        load_library(path, invalid_policy="skip")
    assert len(error.value.rejections) == 1


def test_empty_library_missing_columns_and_directory_raise(tmp_path):
    with pytest.raises(ValueError, match="No CSV files"):
        load_library(tmp_path)
    path = write_library(tmp_path, [])
    with pytest.raises(LibraryValidationError, match="No valid compounds"):
        load_library(path)
    path.write_text("Catalog ID,smiles\na,CCO\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_library(path)


def test_stereoisomers_remain_distinct(tmp_path):
    path = write_library(tmp_path, [("r", "N[C@@H](C)C(=O)O", -5), ("s", "N[C@H](C)C(=O)O", -6)])
    library, rejected = load_library(path)
    assert len(library) == 2
    assert rejected.empty
    features = featurize(library.smiles)
    assert not np.array_equal(features[0], features[1])


def test_live_library_ignores_optional_truth_and_metadata(tmp_path):
    path = tmp_path / "library.csv"
    path.write_text("id,structure,leaky_score\na,CCO,-12\n")
    library, _ = load_library(path, "id", "structure", score_column=None)
    assert list(library.columns) == ["compound_id", "smiles", "source_file", "source_row"]
    assert featurize(library.smiles).shape == (1, 2048)


def test_deduplication_can_preserve_different_ids_but_not_same_id(tmp_path):
    path = write_library(tmp_path, [("a", "CCO", -5), ("b", "OCC", -5), ("a", "CCO", -5)])
    library, rejected = load_library(path, deduplicate=False)
    assert library["compound_id"].tolist() == ["a", "b"]
    assert rejected.reason.tolist() == ["duplicate_compound_id"]


def test_repeated_duplicate_alias_report_points_to_actual_retained_compound(tmp_path):
    path = write_library(tmp_path, [("a", "CCO", -5), ("b", "OCC", -5), ("b", "CCO", -5)])
    library, rejected = load_library(path)
    assert library.compound_id.tolist() == ["a"]
    assert rejected.kept_compound_id.tolist() == ["a", "a"]


@pytest.mark.parametrize("invalid_policy", ["error", "skip"])
@pytest.mark.parametrize("contents,record_number,message", [
    ("Catalog ID,smiles,docking_score\na,CCO,-5,extra\n", 2, "field count"),
    ("Catalog ID,smiles,docking_score\na,CCO\n", 2, "field count"),
    ("Catalog ID,smiles,docking_score\na,CCO,-5\n\n", 3, "field count"),
    ("Catalog ID,smiles,docking_score,smiles\na,CCO,-5,CCN\n", 1, "Duplicate"),
    ("Catalog ID,smiles,docking_score,meta,meta\na,CCO,-5,x,y\n", 1, "Duplicate"),
    ('Catalog ID,smiles,docking_score\n"a,CCO,-5\n', 2, "Malformed"),
    ('Catalog ID,smiles,docking_score\n"a"oops,CCO,-5\n', 2, "Malformed"),
    ('Catalog ID,smiles,docking_score,meta\na,CCO,-5,"line1\nline2"\nb,CCN,-4\n', 3, "field count"),
])
def test_csv_structure_errors_are_fatal_with_logical_record_location(
    tmp_path, invalid_policy, contents, record_number, message,
):
    path = tmp_path / "malformed.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message) as error:
        load_library(path, invalid_policy=invalid_policy)
    assert f"{path}:{record_number}" in str(error.value)


def test_csv_bom_quoted_fields_and_metadata_preserve_logical_rows(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('Catalog ID,smiles,docking_score,note\n'
                    '"009,alpha",OCC,-5,"first line\nsecond, line"\n'
                    '"NA\nbeta",CCN,-4,"quoted ""text"""\n'
                    'bad,invalid,-3,ignored\nlast,CC,-2,\n', encoding="utf-8-sig")
    library, rejected = load_library(path, invalid_policy="skip")
    assert library.compound_id.tolist() == ["009,alpha", "NA\nbeta", "last"]
    assert library.smiles.tolist() == ["CCO", "CCN", "CC"]
    assert library.score.tolist() == [-5, -4, -2]
    assert library.source_row.tolist() == [2, 3, 5]
    assert rejected.source_row.tolist() == [4]


def test_duplicate_representatives_and_seeded_selection_ignore_file_and_row_order(tmp_path):
    rows = [("z", "OCC", -5), ("b", "CCO", -5), ("b", "OCC", -5), ("a", "CCO", -5),
            ("c", "CC", -3), ("d", "CCN", -4), ("e", "CCC", -2)]
    baseline = None
    for arrangement, order in enumerate((list(range(7)), list(reversed(range(7))), [2, 4, 0, 5, 1, 6, 3])):
        directory = tmp_path / str(arrangement)
        directory.mkdir()
        ordered = [rows[index] for index in order]
        write_library(directory, ordered[:3], "a.csv")
        write_library(directory, ordered[3:], "z.csv")
        library, rejected = load_library(directory)
        assert library.compound_id.tolist() == ["a", "c", "d", "e"]
        representative = library.iloc[0]
        source = pd.read_csv(representative.source_file).iloc[representative.source_row - 2]
        assert source["Catalog ID"] == "a"
        assert rejected.kept_compound_id.tolist() == ["a"] * 3
        assert sorted(rejected.reason) == ["duplicate_compound_id", "duplicate_structure", "duplicate_structure"]
        assert sorted(rejected.compound_id) == ["b", "b", "z"]
        for record in rejected.itertuples(index=False):
            source = pd.read_csv(record.source_file).iloc[record.source_row - 2]
            assert source["Catalog ID"] == record.compound_id
        selected = select_indices(library.score, None, 3, strategy="random",
                                  rng=np.random.default_rng(73), ids=library.compound_id)
        result = library.compound_id.iloc[selected].tolist()
        if baseline is None:
            baseline = result
        assert result == baseline


@pytest.mark.parametrize("rows,reason", [
    ([("z", "CCO", -5), ("a", "OCC", -5), ("z", "CCN", -5)], "conflicting_compound_id"),
    ([("z", "CCO", -5), ("a", "OCC", -5), ("b", "OCC", -6)], "conflicting_structure_score"),
])
def test_conflicts_remain_fatal_after_representative_changes(tmp_path, rows, reason):
    with pytest.raises(LibraryValidationError, match=reason):
        load_library(write_library(tmp_path, rows), invalid_policy="skip")


def test_fatal_rejections_also_point_to_last_valid_representative(tmp_path):
    rows = [("z", "CCO", -5), ("b", "OCC", -5), ("b", "CCO", -5), ("a", "OCC", -5), ("z", "CCN", -5)]
    with pytest.raises(LibraryValidationError) as error:
        load_library(write_library(tmp_path, rows), invalid_policy="skip")
    assert error.value.rejections.kept_compound_id.tolist() == ["a"] * 4


def test_fingerprints_deterministic_binary_and_canonical_invariant():
    first = featurize(["CCO", "OCC", "c1ccccc1"])
    second = featurize(["CCO", "OCC", "c1ccccc1"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], first[1])
    assert first.dtype == np.uint8
    assert first.shape == (3, 2048)
    assert set(np.unique(first)) == {0, 1}
    assert first.sum(axis=1).min() > 0
    assert featurize([], n_bits=128).shape == (0, 128)
    with pytest.raises(ValueError, match="Invalid SMILES"):
        featurize(["invalid"])
    with pytest.raises(ValueError, match="not one string"):
        featurize("CCO")


def test_feature_morgan_uses_pharmacophoric_invariants():
    smiles = ["CCO", "CCS", "c1ccccc1O"]
    connectivity = featurize(smiles, n_bits=512)
    features = featurize(smiles, n_bits=512, fingerprint_type="feature_morgan")
    assert features.dtype == np.uint8 and features.shape == connectivity.shape
    assert set(np.unique(features)) == {0, 1}
    assert not np.array_equal(connectivity, features)
    # Feature invariants treat the hydroxyl and thiol donors alike at radius 0.
    zero = featurize(smiles[:2], n_bits=512, radius=0, fingerprint_type="feature_morgan")
    np.testing.assert_array_equal(zero[0], zero[1])
    np.testing.assert_array_equal(features, featurize(smiles, n_bits=512, fingerprint_type="feature_morgan"))


@pytest.mark.parametrize("kwargs", [{"radius": -1}, {"radius": True}, {"n_bits": 0}, {"n_bits": 2.5},
                                    {"fingerprint_type": "maccs"}])
def test_invalid_fingerprint_configuration(kwargs):
    with pytest.raises(ValueError):
        featurize(["CCO"], **kwargs)


def test_random_holdout_reproducible_complete_and_groups_duplicates():
    smiles = ["CCO", "OCC", "CCN", "CC", "C", "CO"]
    candidate, holdout = split_holdout(smiles, 0.3, seed=73)
    repeated = split_holdout(smiles, 0.3, seed=73)
    np.testing.assert_array_equal(candidate, repeated[0])
    np.testing.assert_array_equal(holdout, repeated[1])
    assert bool(0 in holdout) == bool(1 in holdout)
    assert not set(candidate) & set(holdout)
    assert sorted(np.r_[candidate, holdout]) == list(range(len(smiles)))


def test_scaffold_holdout_has_no_scaffold_overlap():
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Oc1ccncc1", "CCO", "CCN"]
    candidates, holdout = split_holdout(smiles, 0.3, seed=12, strategy="scaffold")
    keys = scaffold_keys(smiles)
    assert not {keys[i] for i in candidates} & {keys[i] for i in holdout}
    assert len(candidates) and len(holdout)
    assert sorted(np.r_[candidates, holdout]) == list(range(len(smiles)))
    assert keys[-2:] == ["", ""]


def test_single_scaffold_is_not_silently_split():
    with pytest.raises(ValueError, match="at least two distinct"):
        split_holdout(["CCO", "CCN"], 0.3, seed=1, strategy="scaffold")
    candidate, holdout = split_holdout(["CCO"], 0, seed=1)
    assert candidate.tolist() == [0]
    assert not len(holdout)


@pytest.mark.parametrize("fraction", [-0.1, 1, float("nan")])
def test_invalid_holdout_fraction(fraction):
    with pytest.raises(ValueError, match="holdout fraction"):
        split_holdout(["CCO", "CCN"], fraction, seed=1)
