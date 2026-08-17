"""
Export and import FER documents.

`FER <https://github.com/jongablop/fer>`_ is a JSON structure for experimental
results.  Its core type, ``Measurement``, is recursive: a measurement's
``Source`` lists the ``input_quantities`` it was computed from, which are
themselves ``Measurement`` objects.  That maps directly onto the ft4ftirs
processing chain, so a spectrum can be written together with the provenance
that produced it:

.. code-block:: text

    averaged spectrum (Measurement)
    └── source = "average_spectra"
        └── input_quantities
            ├── forward spectrum (Measurement)
            │   └── source = "SpectralPipeline"
            │       └── input_quantities -> forward interferogram
            │                                └── source = the spectrometer
            └── backward spectrum (Measurement)
                └── ...

Conventions
-----------
* **Source** carries what produced the value: the model, and every quantity
  that propagates into the result.  **State** carries conditions and
  categorical choices that do not.
* ``standard_uncertainties`` are written as **empty arrays**.  ft4ftirs does
  not compute uncertainties, and writing zeros would assert perfect knowledge
  in a GUM-shaped field.
* Arrays are stored as arrays, and categorical settings as strings, so a
  document can be checked without a ft4ftirs lookup table.

``ferpy`` is an optional dependency; it is imported only when these functions
are called.  Install it with ``pip install ft4ftirs[fer]``.

Notes
-----
:func:`write_fer` normalises ``null`` list-valued fields to ``[]`` on the way
out, and :func:`read_fer` tolerates them on the way in.  This works around a
``ferpy`` round-trip bug: ``to_dict()`` writes ``null`` for empty optional
collections while ``from_dict()`` reads them with ``data.get(key, [])``, which
returns ``None`` when the key is present and null, and then iterates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ft4ftirs.data.interferogram import Interferogram, ScanDirection
from ft4ftirs.data.spectrum import SpectralQuantity, Spectrum

__all__ = [
    "FerDocument",
    "write_fer",
    "read_fer",
    "interferogram_measurement",
    "spectrum_measurement",
    "averaged_measurement",
]


# ===========================================================================
# Vocabulary — the names written into the document
# ===========================================================================

_Q_SAMPLE_INDEX = "Sample index"
_Q_INTERFEROGRAM = "Interferogram signal"
_Q_WAVENUMBER = "Wavenumber"
_Q_INTENSITY = "Spectral intensity"

#: influence-quantity name -> key used in ``Interferogram.metadata`` /
#: ``Spectrum.metadata`` when the document is read back.
_SCALAR_TO_METADATA = {
    "Laser wavenumber": "laser_wavenumber_cm",
    "Common scaling factor": "csf",
    "Signal gain": "signal_gain",
    "Zero-filling factor": "zero_filling_factor",
    "Maximum retardation": "max_retardation_points",
    "FFT size": "fft_size",
    "Phase resolution": "phase_resolution_cm",
    "ZPD position": "zpd_index",
    "Number of spectra averaged": "n_spectra_averaged",
}

#: State name -> metadata key.
_STATE_TO_METADATA = {
    "Scan direction": "scan_direction",
    "OPUS block": "opus_block",
    "Apodization window": "apodization_window",
    "Phase correction": "phase_corrector",
    "ZPD finder": "zpd_finder",
    "Spectral quantity": "quantity",
}

#: Separator between a State's value and its explanatory note.  The value is
#: everything before the first occurrence, so states survive a round trip.
_NOTE_SEP = " — "

_LIST_KEYS = frozenset(
    {
        "changelog",
        "state",
        "coverages",
        "probability_density_functions",
        "influence_quantities",
        "input_quantities",
        "results",
    }
)

_PIPELINE_MODEL = (
    r"$S(\tilde{\nu}) = \Re\{\mathcal{F}[W(x)\,I(x)]\,"
    r"e^{-i\varphi(\tilde{\nu})}\}$"
)
_AVERAGE_MODEL = r"$\bar{S}(\tilde{\nu}) = \frac{1}{N}\sum_i S_i(\tilde{\nu})$"


# ===========================================================================
# ferpy access
# ===========================================================================


def _ferpy():
    """Import ferpy lazily and return the classes used here."""
    try:
        from ferpy.auxiliary.changelog_entry import ChangelogEntry
        from ferpy.auxiliary.state import State
        from ferpy.main.measurement import Measurement
        from ferpy.main.quantity_values import QuantityValues
        from ferpy.main.source import Source
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "FER support requires the 'ferpy' package, which is an optional "
            "dependency of ft4ftirs. Install it with: pip install ferpy"
        ) from exc
    return Measurement, QuantityValues, Source, State, ChangelogEntry


# ===========================================================================
# Small builders
# ===========================================================================


def _qv(name, quantities, symbols, units, values, description=None):
    """QuantityValues with empty (= not computed) standard uncertainties."""
    _, QuantityValues, _, _, _ = _ferpy()
    return QuantityValues(
        name=name,
        description=description,
        quantities=list(quantities),
        symbols=list(symbols),
        units=list(units),
        values=[np.asarray(v, dtype=float).tolist() for v in values],
        standard_uncertainties=[[] for _ in values],
    )


def _scalar(name, quantity, symbol, unit, value, description=None):
    """A single-valued QuantityValues."""
    return _qv(name, [quantity], [symbol], [unit], [[float(value)]], description)


def _state(name, value, note=None):
    """A categorical setting: a FER State with no quantity_value."""
    _, _, _, State, _ = _ferpy()
    text = f"{value}" if note is None else f"{value}{_NOTE_SEP}{note}"
    return State(name=name, description=text)


def _entry(text):
    _, _, _, _, ChangelogEntry = _ferpy()
    return ChangelogEntry(
        timestamp=datetime.now(timezone.utc), user="ft4ftirs", description=text
    )


def _version() -> str:
    from ft4ftirs import __version__

    return __version__


# ===========================================================================
# Export
# ===========================================================================


def interferogram_measurement(
    interferogram: Interferogram,
    *,
    instrument_name: Optional[str] = None,
    instrument_description: Optional[str] = None,
):
    """
    Build a FER ``Measurement`` for one interferogram.

    The interferogram's ``Source`` is the spectrometer: a terminal node with no
    ``input_quantities``.

    Parameters
    ----------
    interferogram : Interferogram
        The interferogram to describe.  Its own attributes and ``metadata`` are
        used — nothing is taken from any other interferogram.
    instrument_name : str, optional
        Name for the spectrometer ``Source``.  Defaults to a generic label
        built from ``metadata['instrument']`` when present.
    instrument_description : str, optional
        Free-text description of the acquisition conditions.

    Returns
    -------
    ferpy.main.measurement.Measurement
    """
    Measurement, _, Source, _, _ = _ferpy()
    meta = interferogram.metadata

    influences = [
        _scalar(
            "Laser wavenumber",
            "Wavenumber",
            r"$\tilde{\nu}_L$",
            "cm^-1",
            interferogram.laser_wavenumber,
            "HeNe reference laser; sampling is every zero crossing, so "
            "dx = 1/(2 * laser wavenumber)",
        )
    ]
    if "csf" in meta:
        influences.append(
            _scalar(
                "Common scaling factor",
                "Scaling factor",
                "$CSF$",
                "1",
                meta["csf"],
                "scaling applied to the raw detector counts",
            )
        )
    if "signal_gain" in meta:
        influences.append(
            _scalar("Signal gain", "Gain", "$ASG$", "1", meta["signal_gain"])
        )

    if instrument_name is None:
        instrument_name = str(meta.get("instrument", "FT-IR spectrometer"))

    source = Source(
        name=instrument_name,
        description=instrument_description,
        model="",  # a physical instrument, not an equation
        influence_quantities=influences,
        input_quantities=[],  # terminal node
    )

    states = [_state("Scan direction", interferogram.scan_direction.value)]
    if "opus_block" in meta:
        states.append(
            _state("OPUS block", meta["opus_block"], "sample vs reference beam")
        )

    origin = meta.get("source_file")
    description = f"{interferogram.scan_direction.value.capitalize()} interferogram"
    if origin:
        description += f" from {Path(origin).name}"

    return Measurement(
        description=description,
        correct=True,
        changelog=[_entry(f"Read with ft4ftirs {_version()}")],
        state=states,
        results=[
            _qv(
                "Interferogram",
                [_Q_SAMPLE_INDEX, _Q_INTERFEROGRAM],
                ["$n$", "$I$"],
                ["1", "a.u."],
                [interferogram.x_index, interferogram.signal],
                "as transformed: ZPD-centred, and time-reversed for a backward scan",
            )
        ],
        measurands=[_Q_INTERFEROGRAM],
        source=source,
    )


def spectrum_measurement(
    spectrum: Spectrum,
    interferogram: Interferogram,
    interferogram_meas=None,
    *,
    apodizer=None,
    phase_corrector=None,
    applied_window: Optional[np.ndarray] = None,
    applied_phase: Optional[np.ndarray] = None,
    instrument_name: Optional[str] = None,
    instrument_description: Optional[str] = None,
):
    """
    Build a FER ``Measurement`` for one transformed spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        The single-beam spectrum produced from ``interferogram``.
    interferogram : Interferogram
        The interferogram it was computed from.
    interferogram_meas : Measurement, optional
        An already-built measurement for ``interferogram``.  Built here when
        omitted.
    apodizer : Apodizer, optional
    phase_corrector : PhaseCorrector, optional
        Recorded by name in ``State``.  When ``applied_window`` /
        ``applied_phase`` are not supplied, these are used to *reconstruct* the
        arrays that were applied; the reconstruction is flagged as such in the
        stored description.
    applied_window, applied_phase : np.ndarray, optional
        The arrays actually applied during processing.  Prefer passing these:
        they are recorded verbatim and need no assumption about how the
        pipeline ran.

    Returns
    -------
    ferpy.main.measurement.Measurement
    """
    Measurement, _, Source, _, _ = _ferpy()

    n_pts = interferogram.n_points
    zpd = interferogram.zpd_index
    wing = max(zpd, n_pts - 1 - zpd)
    meta = dict(spectrum.metadata)
    fft_size = meta.get("fft_size")

    if interferogram_meas is None:
        interferogram_meas = interferogram_measurement(
            interferogram,
            instrument_name=instrument_name,
            instrument_description=instrument_description,
        )

    influences = [
        _scalar(
            "Laser wavenumber",
            "Wavenumber",
            r"$\tilde{\nu}_L$",
            "cm^-1",
            interferogram.laser_wavenumber,
            "sets the wavenumber axis",
        ),
        _scalar(
            "Maximum retardation",
            "Number of points",
            r"$n_L$",
            "1",
            wing,
            "samples from the ZPD to the far end of the record; the resolution "
            "element is 1/(n_L * dx)",
        ),
        _scalar("ZPD position", "Sample index", "$n_{ZPD}$", "1", zpd),
    ]
    if "zero_filling_factor" in meta:
        influences.append(
            _scalar(
                "Zero-filling factor",
                "Zero-filling factor",
                "$Z$",
                "1",
                meta["zero_filling_factor"],
                "spectral points per resolution element, applied to the "
                "maximum retardation",
            )
        )
    if fft_size is not None:
        influences.append(
            _scalar(
                "FFT size",
                "Number of points",
                "$N$",
                "1",
                fft_size,
                "fixes the spectral point spacing",
            )
        )
    phase_resolution = interferogram.metadata.get("phase_resolution_cm")
    if phase_resolution is not None:
        influences.append(
            _scalar(
                "Phase resolution",
                "Wavenumber",
                r"$\Delta\tilde{\nu}_\varphi$",
                "cm^-1",
                phase_resolution,
            )
        )

    # --- the arrays that were applied ------------------------------------
    reconstructed = []
    window = applied_window
    if window is None and apodizer is not None:
        window = np.roll(apodizer.window(n_pts), zpd - n_pts // 2)
        reconstructed.append("window")
    if window is not None:
        influences.append(
            _qv(
                "Applied apodization window",
                [_Q_SAMPLE_INDEX, "Window value"],
                ["$n$", "$W$"],
                ["1", "1"],
                [np.arange(n_pts), window],
                "the ZPD-aligned window multiplied into the interferogram"
                + (" (reconstructed at export)" if "window" in reconstructed else ""),
            )
        )

    phase = applied_phase
    if phase is None and phase_corrector is not None and fft_size is not None:
        if hasattr(phase_corrector, "estimate_phase") and window is not None:
            phase = phase_corrector.estimate_phase(
                complex_spectrum=np.zeros(fft_size // 2, dtype=complex),
                wavenumbers=None,
                zpd_index=zpd,
                apodized_signal=window * interferogram.signal,
            )
            reconstructed.append("phase")
    if phase is not None:
        phase_axis = (
            np.arange(len(phase)) * 2.0 * interferogram.laser_wavenumber / fft_size
        )
        influences.append(
            _qv(
                "Applied phase",
                [_Q_WAVENUMBER, "Phase"],
                [r"$\tilde{\nu}$", r"$\varphi$"],
                ["cm^-1", "rad"],
                [phase_axis, phase],
                "phase specific to this scan direction"
                + (" (reconstructed at export)" if "phase" in reconstructed else ""),
            )
        )

    window_name = meta.get("apodization_window")
    if window_name is None and apodizer is not None:
        window_name = getattr(apodizer.window, "name", type(apodizer.window).__name__)
    corrector_name = meta.get("phase_corrector")
    if corrector_name is None and phase_corrector is not None:
        corrector_name = type(phase_corrector).__name__

    bits = []
    if window_name:
        bits.append(f"{window_name} apodization")
    if fft_size is not None:
        bits.append(f"transform size {fft_size}")
    if corrector_name:
        bits.append(str(corrector_name))
    if reconstructed:
        bits.append(f"{' and '.join(reconstructed)} reconstructed at export")

    source = Source(
        name=f"ft4ftirs SpectralPipeline v{_version()}",
        description=", ".join(bits) if bits else None,
        model=_PIPELINE_MODEL,
        influence_quantities=influences,
        input_quantities=[interferogram_meas],
    )

    states = [_state("Scan direction", interferogram.scan_direction.value)]
    if window_name:
        states.append(_state("Apodization window", window_name))
    if corrector_name:
        states.append(_state("Phase correction", corrector_name))
    if interferogram.zpd_finder is not None:
        states.append(_state("ZPD finder", type(interferogram.zpd_finder).__name__))

    return Measurement(
        description=(
            f"Single-beam spectrum from the {interferogram.scan_direction.value} scan"
        ),
        correct=True,
        changelog=[_entry(f"Processed with ft4ftirs {_version()}")],
        state=states,
        results=[
            _qv(
                "Single-beam spectrum",
                [_Q_WAVENUMBER, _Q_INTENSITY],
                [r"$\tilde{\nu}$", "$S$"],
                ["cm^-1", "a.u."],
                [spectrum.wavenumbers, spectrum.intensities],
            )
        ],
        measurands=[_Q_INTENSITY],
        source=source,
    )


def averaged_measurement(spectrum: Spectrum, components: Sequence):
    """
    Build a FER ``Measurement`` for a spectrum averaged from ``components``.

    Parameters
    ----------
    spectrum : Spectrum
        The averaged spectrum.
    components : sequence of Measurement
        The per-spectrum measurements that were averaged.

    Returns
    -------
    ferpy.main.measurement.Measurement
    """
    Measurement, _, Source, _, _ = _ferpy()
    meta = dict(spectrum.metadata)

    influences = [
        _scalar(
            "Number of spectra averaged",
            "Count",
            "$N$",
            "1",
            meta.get("n_spectra_averaged", len(components)),
        )
    ]

    source = Source(
        name="ft4ftirs.processing.scan_averaging.average_spectra",
        description=(
            "Point-wise mean of the independently phase-corrected spectra. The "
            "interferograms are deliberately NOT averaged: the scan directions "
            "carry different phase errors."
        ),
        model=_AVERAGE_MODEL,
        influence_quantities=influences,
        input_quantities=list(components),
    )

    return Measurement(
        description="Single-beam spectrum, scan directions averaged",
        correct=True,
        changelog=[
            _entry(f"Averaged {len(components)} spectra with ft4ftirs {_version()}")
        ],
        state=[_state("Spectral quantity", spectrum.quantity.name)],
        results=[
            _qv(
                "Averaged single-beam spectrum",
                [_Q_WAVENUMBER, _Q_INTENSITY],
                [r"$\tilde{\nu}$", "$S$"],
                ["cm^-1", "a.u."],
                [spectrum.wavenumbers, spectrum.intensities],
            )
        ],
        measurands=[_Q_INTENSITY],
        source=source,
    )


def _normalise_nulls(obj):
    """Replace null list-valued fields with ``[]`` so ferpy can read them."""
    if isinstance(obj, dict):
        return {
            k: ([] if (v is None and k in _LIST_KEYS) else _normalise_nulls(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalise_nulls(v) for v in obj]
    return obj


def write_fer(
    path: Union[str, Path],
    spectrum: Spectrum,
    *,
    interferograms: Optional[Sequence[Interferogram]] = None,
    components: Optional[Sequence[Spectrum]] = None,
    apodizer=None,
    phase_corrector=None,
    instrument_name: Optional[str] = None,
    instrument_description: Optional[str] = None,
    indent: int = 2,
) -> Path:
    """
    Write ``spectrum`` and its provenance to a FER JSON document.

    Two shapes are supported:

    * **one spectrum from one interferogram** — pass a single interferogram and
      no ``components``;
    * **an averaged spectrum** — pass the per-direction ``components`` and the
      interferograms they came from, in the same order.

    Parameters
    ----------
    path : str or Path
        Destination file.
    spectrum : Spectrum
        The spectrum to store at the root of the document.
    interferograms : sequence of Interferogram, optional
        The interferograms behind the spectrum.  Required to record provenance;
        without them the document has a spectrum and no inputs.
    components : sequence of Spectrum, optional
        Per-direction spectra that were averaged into ``spectrum``.  Must be
        the same length as ``interferograms``.
    apodizer, phase_corrector : optional
        Passed through to :func:`spectrum_measurement`.
    instrument_name, instrument_description : str, optional
        Identify the spectrometer in the terminal Source node.
    indent : int, default 2
        JSON indentation.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If ``components`` and ``interferograms`` have different lengths.
    """
    path = Path(path)
    interferograms = list(interferograms or [])
    components = list(components) if components is not None else None

    if components is not None and len(components) != len(interferograms):
        raise ValueError(
            f"components and interferograms must be the same length; "
            f"got {len(components)} and {len(interferograms)}."
        )

    kwargs = dict(
        apodizer=apodizer,
        phase_corrector=phase_corrector,
        instrument_name=instrument_name,
        instrument_description=instrument_description,
    )

    if components is not None:
        per_direction = [
            spectrum_measurement(s, ig, **kwargs)
            for s, ig in zip(components, interferograms)
        ]
        root = averaged_measurement(spectrum, per_direction)
    elif len(interferograms) == 1:
        root = spectrum_measurement(spectrum, interferograms[0], **kwargs)
    elif len(interferograms) > 1:
        raise ValueError(
            "several interferograms were given without their per-direction "
            "spectra; pass components= as well, or a single interferogram."
        )
    else:
        raise ValueError(
            "at least one interferogram is required to record provenance."
        )

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_normalise_nulls(root.to_dict()), fh, indent=indent)
    return path


# ===========================================================================
# Import
# ===========================================================================


@dataclass
class FerDocument:
    """
    A FER document read back into ft4ftirs objects.

    Attributes
    ----------
    spectrum : Spectrum
        The spectrum at the root of the document.
    components : list of Spectrum
        Per-direction spectra, when the root was an average.  Empty otherwise.
    interferograms : list of Interferogram
        Every interferogram found in the provenance graph, in document order.
    measurement : ferpy.main.measurement.Measurement
        The parsed FER object, for anything this wrapper does not expose.
    """

    spectrum: Spectrum
    components: List[Spectrum] = field(default_factory=list)
    interferograms: List[Interferogram] = field(default_factory=list)
    measurement: Any = None


def _state_value(text: Optional[str]) -> Optional[str]:
    """The value part of a State description, dropping any explanatory note."""
    if text is None:
        return None
    return text.split(_NOTE_SEP)[0].strip()


def _series(qv, quantity_name: str) -> np.ndarray:
    """Pull one named series out of a QuantityValues by quantity name."""
    index = qv.quantities.index(quantity_name)
    return np.asarray(qv.values[index], dtype=float)


def _has_quantities(measurement, *names) -> bool:
    results = measurement.results or []
    if not results:
        return False
    quantities = results[0].quantities or []
    return all(n in quantities for n in names)


def _collect_metadata(measurement) -> Dict[str, Any]:
    """Scalars and states of one Measurement, as a ft4ftirs-style dict."""
    meta: Dict[str, Any] = {}
    for st in measurement.state or []:
        key = _STATE_TO_METADATA.get(st.name)
        if key:
            meta[key] = _state_value(st.description)
    source = measurement.source
    if source is not None:
        for iq in source.influence_quantities or []:
            key = _SCALAR_TO_METADATA.get(iq.name)
            if key and iq.values and len(iq.values) == 1 and len(iq.values[0]) == 1:
                meta[key] = iq.values[0][0]
    return meta


def _interferogram_from(measurement) -> Interferogram:
    qv = measurement.results[0]
    meta = _collect_metadata(measurement)

    laser = meta.get("laser_wavenumber_cm")
    if laser is None:
        raise ValueError(
            f"interferogram '{measurement.description}' carries no laser "
            "wavenumber; cannot rebuild an Interferogram from it."
        )

    direction = meta.get("scan_direction", ScanDirection.UNKNOWN.value)
    try:
        scan_direction = ScanDirection(direction)
    except ValueError:
        scan_direction = ScanDirection.UNKNOWN

    meta["source"] = measurement.source.name if measurement.source else None
    meta["description"] = measurement.description

    return Interferogram(
        signal=_series(qv, _Q_INTERFEROGRAM),
        laser_wavenumber=float(laser),
        x_index=_series(qv, _Q_SAMPLE_INDEX),
        scan_direction=scan_direction,
        metadata=meta,
    )


def _spectrum_from(measurement) -> Spectrum:
    qv = measurement.results[0]
    meta = _collect_metadata(measurement)

    quantity = SpectralQuantity.SINGLE_BEAM
    name = meta.pop("quantity", None)
    if name:
        try:
            quantity = SpectralQuantity[name]
        except KeyError:
            meta["quantity_name"] = name

    meta["description"] = measurement.description
    if measurement.source is not None:
        meta["source"] = measurement.source.name

    return Spectrum(
        wavenumbers=_series(qv, _Q_WAVENUMBER),
        intensities=_series(qv, _Q_INTENSITY),
        quantity=quantity,
        metadata=meta,
    )


def read_fer(path: Union[str, Path]) -> FerDocument:
    """
    Read a FER document written by :func:`write_fer`.

    Null list-valued fields are normalised to ``[]`` before parsing, so
    documents produced by ``ferpy`` itself (which cannot re-read its own
    output) load here as well.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    FerDocument

    Notes
    -----
    A rebuilt :class:`~ft4ftirs.data.interferogram.Interferogram` is
    mean-subtracted again by its constructor.  For data written by this module
    that is a no-op, since the stored signal is already centred.
    """
    Measurement, _, _, _, _ = _ferpy()

    with open(path, "r", encoding="utf-8") as fh:
        data = _normalise_nulls(json.load(fh))
    root = Measurement.from_dict(data)

    if not _has_quantities(root, _Q_WAVENUMBER, _Q_INTENSITY):
        raise ValueError(
            f"{path}: the root measurement does not hold a spectrum "
            f"('{_Q_WAVENUMBER}' and '{_Q_INTENSITY}' results)."
        )

    spectrum = _spectrum_from(root)
    components: List[Spectrum] = []
    interferograms: List[Interferogram] = []

    def walk(measurement) -> None:
        source = measurement.source
        for child in (source.input_quantities or []) if source else []:
            if _has_quantities(child, _Q_SAMPLE_INDEX, _Q_INTERFEROGRAM):
                interferograms.append(_interferogram_from(child))
            elif _has_quantities(child, _Q_WAVENUMBER, _Q_INTENSITY):
                components.append(_spectrum_from(child))
            walk(child)

    walk(root)

    return FerDocument(
        spectrum=spectrum,
        components=components,
        interferograms=interferograms,
        measurement=root,
    )
