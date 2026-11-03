"""Read label candidates from ``specObj-dr17.fits`` and apply the ADR 0002 basic filters.

Only the needed columns are read, one at a time, through a memory map: the file has 133
columns and 5.8 million rows, so loading it whole would not fit in memory. Missing values are
stored as NULL, never as NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from astropy.io import fits

__all__ = [
    "ALLOWED_CLASSES",
    "DEFAULT_COLUMNS",
    "apply_basic_filters",
    "load_candidates",
]

DEFAULT_COLUMNS: Final[tuple[str, ...]] = (
    "SPECOBJID",
    "PLATEID",
    "PLATE",
    "MJD",
    "FIBERID",
    "CLASS",
    "SUBCLASS",
    "Z",
    "Z_ERR",
    "ZWARNING",
    "PLUG_RA",
    "PLUG_DEC",
    "SPECPRIMARY",
    "PRIMTARGET",
    "SURVEY",
    "PROGRAMNAME",
)
ALLOWED_CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")

_FLOAT_KINDS: Final = "f"
_STRING_KINDS: Final = "SU"
# FITS fixed-width strings are padded with spaces, and numpy may leave NUL bytes.
_STRIP_CHARS: Final = " \x00"


def _native(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Return a C-contiguous copy in native byte order (FITS data is big-endian)."""
    return np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("="))


def _to_arrow(values: npt.NDArray[np.generic], name: str) -> pa.Array:
    kind = values.dtype.kind
    if kind in _STRING_KINDS:
        text = pa.array(np.char.decode(values, "ascii") if kind == "S" else values)
        return pc.utf8_trim(text.cast(pa.string()), characters=_STRIP_CHARS)
    native = _native(values)
    if kind in _FLOAT_KINDS:
        return pa.array(native, mask=np.isnan(native))
    if kind in "iu":
        return pa.array(native)
    raise ValueError(f"column {name} has unsupported dtype {values.dtype}")


def load_candidates(
    fits_path: Path | str,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pa.Table:
    """Read ``columns`` from the first table HDU of ``fits_path``; apply no filter.

    Names are lower-cased. Fixed-width strings become stripped ``str`` (``SPECOBJID`` is
    kept as a string), and NaN in floating point columns becomes NULL.
    """
    path = Path(fits_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FITS file not found: {path.name}; download specObj-dr17.fits first"
        )
    arrays: list[pa.Array] = []
    names: list[str] = []
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        table_hdu = hdul[1]
        available = {str(name).upper() for name in table_hdu.columns.names}
        missing = [name for name in columns if name.upper() not in available]
        if missing:
            raise KeyError(f"columns not in {path.name}: {', '.join(missing)}")
        for column in columns:
            raw = np.asarray(table_hdu.data[column])
            arrays.append(_to_arrow(raw, column))
            names.append(column.lower())
            del raw
    return pa.Table.from_arrays(arrays, names=names)


def apply_basic_filters(table: pa.Table) -> tuple[pa.Table, dict[str, int]]:
    """Apply the ADR 0002 filters in order and count what each one discards.

    The report maps ``input_rows`` and one ``dropped_*`` key per filter (counted on the rows
    that survived the previous filters) to counts, then ``kept``. A NULL test value counts
    as a failure of that filter.
    """
    report: dict[str, int] = {"input_rows": table.num_rows}

    def run(key: str, condition: pa.ChunkedArray | pa.Array) -> None:
        nonlocal table
        keep = pc.fill_null(condition, False)
        report[key] = table.num_rows - int(pc.sum(keep.cast(pa.int64())).as_py() or 0)
        table = table.filter(keep)

    run("dropped_specprimary_not_1", pc.equal(table["specprimary"], 1))
    run("dropped_zwarning_nonzero", pc.equal(table["zwarning"], 0))
    run(
        "dropped_class_not_allowed",
        pc.is_in(table["class"], value_set=pa.array(ALLOWED_CLASSES)),
    )
    run(
        "dropped_invalid_coordinates",
        pc.and_(
            pc.and_(
                pc.is_finite(table["plug_ra"]),
                pc.is_finite(table["plug_dec"]),
            ),
            pc.less_equal(pc.abs(table["plug_dec"]), 90.0),
        ),
    )
    report["kept"] = table.num_rows
    return table, report
