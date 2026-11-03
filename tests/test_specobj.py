"""Offline tests for astro_triage.labels.specobj on a small synthetic FITS file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from astro_triage.labels import specobj
from astro_triage.labels.specobj import (
    ALLOWED_CLASSES,
    DEFAULT_COLUMNS,
    apply_basic_filters,
    load_candidates,
)

BIG_ID = "506711614938966016"  # 19 digits: not exactly representable as float64
_NAN = float("nan")

# (class, z, zwarning, specprimary, ra, dec, first filter that discards it)
ROWS = [
    ("STAR  ", 0.1, 0, 1, 10.0, 5.0, "keep"),
    ("GALAXY", 0.2, 0, 0, 11.0, 6.0, "specprimary"),
    ("QSO\0\0\0", 1.5, 4, 1, 12.0, 7.0, "zwarning"),
    ("SKY   ", 0.0, 0, 1, 13.0, 8.0, "class"),
    ("STAR  ", 0.3, 0, 1, 14.0, 91.0, "coordinates"),
    ("GALAXY", 0.4, 0, 1, _NAN, 9.0, "coordinates"),
    ("QSO   ", _NAN, 0, 1, 15.0, -90.0, "keep"),
    (
        "SKY   ",
        0.0,
        4,
        0,
        16.0,
        1.0,
        "specprimary",
    ),  # fails three filters, first counts
    ("FOO   ", 0.0, 4, 1, 17.0, 2.0, "zwarning"),  # fails zwarning and class
    ("STAR  ", 0.5, 0, 1, 18.0, _NAN, "coordinates"),
]
# Computed by hand from ROWS.
EXPECTED = {
    "input_rows": 10,
    "dropped_specprimary_not_1": 2,
    "dropped_zwarning_nonzero": 2,
    "dropped_class_not_allowed": 1,
    "dropped_invalid_coordinates": 3,
    "kept": 2,
}


def _write_fits(path: Path) -> Path:
    n = len(ROWS)
    ids = [f"{int(BIG_ID) + i}".ljust(22) for i in range(n)]
    cols = [
        fits.Column("SPECOBJID", "22A", array=np.array(ids)),
        fits.Column("PLATEID", "J", array=np.arange(n, dtype=">i4") + 266),
        fits.Column("PLATE", "J", array=np.arange(n, dtype=">i4") + 300),
        fits.Column("MJD", "J", array=np.full(n, 51602, dtype=">i4")),
        fits.Column("FIBERID", "J", array=np.arange(n, dtype=">i4") + 1),
        fits.Column("CLASS", "6A", array=np.array([r[0] for r in ROWS])),
        fits.Column("SUBCLASS", "21A", array=np.array(["  BROADLINE  "] * n)),
        fits.Column("Z", "E", array=np.array([r[1] for r in ROWS], dtype=">f4")),
        fits.Column("Z_ERR", "E", array=np.full(n, 1e-5, dtype=">f4")),
        fits.Column("ZWARNING", "J", array=np.array([r[2] for r in ROWS], dtype=">i4")),
        fits.Column("PLUG_RA", "D", array=np.array([r[4] for r in ROWS], dtype=">f8")),
        fits.Column("PLUG_DEC", "D", array=np.array([r[5] for r in ROWS], dtype=">f8")),
        fits.Column(
            "SPECPRIMARY", "B", array=np.array([r[3] for r in ROWS], dtype="u1")
        ),
        fits.Column("PRIMTARGET", "J", array=np.zeros(n, dtype=">i4")),
        fits.Column("SURVEY", "8A", array=np.array(["sdss   "] * n)),
        fits.Column("PROGRAMNAME", "16A", array=np.array(["legacy "] * n)),
        fits.Column("EXTRA_UNUSED", "D", array=np.zeros(n, dtype=">f8")),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return path


@pytest.fixture
def fits_path(tmp_path: Path) -> Path:
    return _write_fits(tmp_path / "specobj_small.fits")


# ------------------------------------------------------------------ load_candidates
def test_fixture_is_big_endian_on_disk(fits_path: Path) -> None:
    with fits.open(fits_path) as hdul:
        assert hdul[1].data["PLUG_RA"].dtype == np.dtype(">f8")
        assert hdul[1].data["PLATE"].dtype == np.dtype(">i4")


def test_reads_only_requested_columns_lowercased(fits_path: Path) -> None:
    table = load_candidates(fits_path, columns=("Z", "CLASS", "SPECOBJID"))
    assert table.column_names == ["z", "class", "specobjid"]
    assert table.num_rows == len(ROWS)


def test_default_columns_exclude_unused_fits_columns(fits_path: Path) -> None:
    table = load_candidates(fits_path)
    assert table.column_names == [c.lower() for c in DEFAULT_COLUMNS]
    assert "extra_unused" not in table.column_names


def test_missing_column_raises_keyerror_naming_it(fits_path: Path) -> None:
    with pytest.raises(KeyError, match="NOPE"):
        load_candidates(fits_path, columns=("Z", "NOPE"))


def test_missing_file_message_has_no_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as info:
        load_candidates(tmp_path / "absent.fits")
    assert "absent.fits" in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_values_are_native_endian(fits_path: Path) -> None:
    table = load_candidates(fits_path, columns=("PLUG_RA", "Z", "PLATE"))
    for name in ("plug_ra", "z", "plate"):
        arr = table[name].combine_chunks().to_numpy(zero_copy_only=False)
        assert arr.dtype.isnative, name
    assert table["plate"].to_pylist() == list(range(300, 310))
    assert table["plug_ra"][0].as_py() == 10.0
    assert table["z"][0].as_py() == pytest.approx(0.1, rel=1e-6)


def test_strings_are_stripped_of_spaces_and_nuls(fits_path: Path) -> None:
    table = load_candidates(
        fits_path, columns=("CLASS", "SUBCLASS", "SURVEY", "PROGRAMNAME")
    )
    assert table["class"].to_pylist() == [r[0].strip(" \0") for r in ROWS]
    assert table["class"][2].as_py() == "QSO"
    assert set(table["subclass"].to_pylist()) == {"BROADLINE"}
    assert set(table["survey"].to_pylist()) == {"sdss"}
    assert set(table["programname"].to_pylist()) == {"legacy"}
    assert table.schema.field("class").type == pa.string()


def test_trailing_spaces_and_nuls_are_stripped_from_unicode_arrays() -> None:
    values = np.array([" STAR\x00", "QSO \x00 "], dtype="U8")
    assert specobj._to_arrow(values, "class").to_pylist() == ["STAR", "QSO"]


def test_specobjid_stays_string_with_all_19_digits(fits_path: Path) -> None:
    table = load_candidates(fits_path, columns=("SPECOBJID",))
    assert table.schema.field("specobjid").type == pa.string()
    ids = table["specobjid"].to_pylist()
    assert ids[0] == BIG_ID
    assert ids == [str(int(BIG_ID) + i) for i in range(len(ROWS))]
    # A float round trip would have corrupted the neighbouring id.
    assert str(int(float(ids[1]))) != ids[1]


def test_nan_becomes_null_never_nan(fits_path: Path) -> None:
    table = load_candidates(fits_path, columns=("Z", "PLUG_RA", "PLUG_DEC", "Z_ERR"))
    assert table["z"].null_count == 1
    assert table["plug_ra"].null_count == 1
    assert table["plug_dec"].null_count == 1
    assert table["z_err"].null_count == 0
    assert table["z"][6].as_py() is None
    assert table["plug_ra"][5].as_py() is None
    for name in ("z", "plug_ra", "plug_dec"):
        valid = table[name].drop_null().to_numpy()
        assert not np.isnan(valid).any(), name


def test_nan_to_null_survives_parquet_roundtrip(
    fits_path: Path, tmp_path: Path
) -> None:
    table = load_candidates(fits_path, columns=("Z", "PLUG_RA"))
    pq.write_table(table, tmp_path / "t.parquet")
    back = pq.read_table(tmp_path / "t.parquet")
    assert back["z"].null_count == 1
    assert back["plug_ra"].null_count == 1


def test_load_candidates_applies_no_filter(fits_path: Path) -> None:
    table = load_candidates(fits_path)
    assert table.num_rows == len(ROWS)
    assert "SKY" in table["class"].to_pylist()
    assert 0 in table["specprimary"].to_pylist()


def test_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        specobj._to_arrow(np.array([True, False]), "flag")


# --------------------------------------------------------------- apply_basic_filters
def test_allowed_classes_constant() -> None:
    assert ALLOWED_CLASSES == ("STAR", "GALAXY", "QSO")


def test_filter_report_matches_hand_counts(fits_path: Path) -> None:
    kept, report = apply_basic_filters(load_candidates(fits_path))
    assert report == EXPECTED
    assert kept.num_rows == EXPECTED["kept"]
    # Rows 0 and 6 survive (Z is NULL in row 6 but Z is not a filter criterion).
    assert kept["fiberid"].to_pylist() == [1, 7]
    assert kept["class"].to_pylist() == ["STAR", "QSO"]
    assert kept["z"].null_count == 1


def test_report_input_equals_kept_plus_drops_in_order(fits_path: Path) -> None:
    _, report = apply_basic_filters(load_candidates(fits_path))
    drops = sum(v for k, v in report.items() if k.startswith("dropped_"))
    assert report["input_rows"] == report["kept"] + drops
    assert list(report) == [
        "input_rows",
        "dropped_specprimary_not_1",
        "dropped_zwarning_nonzero",
        "dropped_class_not_allowed",
        "dropped_invalid_coordinates",
        "kept",
    ]


def _one_row(**overrides: pa.Array) -> pa.Table:
    base: dict[str, pa.Array] = {
        "specprimary": pa.array([1], pa.uint8()),
        "zwarning": pa.array([0], pa.int32()),
        "class": pa.array(["STAR"], pa.string()),
        "plug_ra": pa.array([1.0], pa.float64()),
        "plug_dec": pa.array([2.0], pa.float64()),
    }
    base.update(overrides)
    return pa.table(base)


def test_row_failing_every_filter_is_charged_to_the_first_only() -> None:
    table = _one_row(
        specprimary=pa.array([0], pa.uint8()),
        zwarning=pa.array([4], pa.int32()),
        **{"class": pa.array(["SKY"])},
        plug_ra=pa.array([_NAN]),
        plug_dec=pa.array([95.0]),
    )
    _, report = apply_basic_filters(table)
    assert report["dropped_specprimary_not_1"] == 1
    assert report["dropped_zwarning_nonzero"] == 0
    assert report["dropped_class_not_allowed"] == 0
    assert report["dropped_invalid_coordinates"] == 0
    assert report["kept"] == 0


def test_baseline_row_is_kept() -> None:
    kept, report = apply_basic_filters(_one_row())
    assert kept.num_rows == 1 and report["kept"] == 1


@pytest.mark.parametrize(
    ("column", "null_array", "key"),
    [
        ("specprimary", pa.array([None], pa.uint8()), "dropped_specprimary_not_1"),
        ("zwarning", pa.array([None], pa.int32()), "dropped_zwarning_nonzero"),
        ("class", pa.array([None], pa.string()), "dropped_class_not_allowed"),
        ("plug_ra", pa.array([None], pa.float64()), "dropped_invalid_coordinates"),
        ("plug_dec", pa.array([None], pa.float64()), "dropped_invalid_coordinates"),
    ],
)
def test_null_in_tested_field_counts_as_filter_failure(
    column: str, null_array: pa.Array, key: str
) -> None:
    kept, report = apply_basic_filters(_one_row(**{column: null_array}))
    assert kept.num_rows == 0
    assert report[key] == 1
    assert report["kept"] == 0


@pytest.mark.parametrize("dec", [90.0, -90.0, 0.0])
def test_dec_boundary_is_inclusive(dec: float) -> None:
    kept, _ = apply_basic_filters(_one_row(plug_dec=pa.array([dec])))
    assert kept.num_rows == 1


@pytest.mark.parametrize("dec", [90.0001, -90.0001, float("inf")])
def test_dec_out_of_range_dropped(dec: float) -> None:
    kept, report = apply_basic_filters(_one_row(plug_dec=pa.array([dec])))
    assert kept.num_rows == 0
    assert report["dropped_invalid_coordinates"] == 1


def test_infinite_ra_dropped() -> None:
    kept, _ = apply_basic_filters(_one_row(plug_ra=pa.array([float("inf")])))
    assert kept.num_rows == 0


@pytest.mark.parametrize("cls", ["star", "SKY", "UNKNOWN", "", "STAR "])
def test_class_match_is_exact(cls: str) -> None:
    kept, report = apply_basic_filters(_one_row(**{"class": pa.array([cls])}))
    assert kept.num_rows == 0
    assert report["dropped_class_not_allowed"] == 1


def test_empty_table_gives_zero_report() -> None:
    kept, report = apply_basic_filters(_one_row().slice(0, 0))
    assert kept.num_rows == 0
    assert all(v == 0 for v in report.values())


def test_all_rows_kept_reports_zero_drops() -> None:
    kept, report = apply_basic_filters(pa.concat_tables([_one_row()] * 3))
    assert kept.num_rows == 3
    assert report["input_rows"] == 3 and report["kept"] == 3
