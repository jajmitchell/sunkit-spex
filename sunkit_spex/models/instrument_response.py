"""Module for model components required for instrument response models."""

import numpy as np

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.units import Quantity

__all__ = ["MatrixModel"]

PHOTON_FLUX_UNIT = u.ph / (u.keV * u.s * u.cm**2)


class MatrixModel(Fittable1DModel):
    r"""
    Fold a photon flux spectrum through a spectral response matrix into counts.

    The model takes a photon flux, :math:`F_i` in ph keV\ :sup:`-1` s\ :sup:`-1`
    cm\ :sup:`-2`, on photon energy bins :math:`i` and returns the counts in
    each count energy bin :math:`j`:

    .. math::

        C_j = c \sum_i F_i \, \Delta E_i \, t \, A \, R_{ij}

    where :math:`\Delta E_i` are the photon bin widths, :math:`t` the exposure
    time, :math:`A` the area the response is normalised to, :math:`R_{ij}` the
    response matrix in counts per photon and :math:`c` the ``conversion_factor``,
    a dimensionless scaling of the response (1 by default, fixed).

    A ``MatrixModel`` created without a ``matrix`` acts as a placeholder in a
    compound model. `~sunkit_spex.fitting.fit_manager.FitManager` fills it in
    from each spectrum's metadata, so one compound model can be fitted to
    several spectra or instruments.

    Units are checked whenever a value is set: any units equivalent to the
    ones below are accepted and converted. Plain numbers are taken to be in
    these units. When evaluated without units (as Astropy's unit-aware
    fitters do internally) the input is taken to be in ph keV\ :sup:`-1`
    s\ :sup:`-1` cm\ :sup:`-2` and the output is in counts.

    The input is always a photon flux: evaluating needs ``photon_edges``,
    ``area`` and ``exposure_time`` as well as the ``matrix``.

    Parameters
    ----------
    matrix : array-like or `~astropy.units.Quantity`, optional
        Response matrix with shape ``(n_photon_bins, n_count_bins)``, in counts
        per photon: dimensionless or in units equivalent to ct/ph.
    photon_edges : `~astropy.units.Quantity`, optional
        Photon energy bin edges (keV), length ``n_photon_bins + 1``.
    area : `~astropy.units.Quantity`, optional
        Area the response is normalised to (cm\ :sup:`2`). A scalar, or one
        value per photon bin for an energy-dependent effective area.
    exposure_time : `~astropy.units.Quantity`, optional
        Exposure (live) time (s).
    conversion_factor : `float`, optional
        Dimensionless scaling of the response, e.g. a calibration or
        efficiency correction. Fixed by default.
    **kwargs :
        Passed to `~astropy.modeling.Fittable1DModel`.

    Notes
    -----
    The output (count bins) generally differs in length from the input
    (photon bins), so the model sets ``verify_dims_in_fitting = False``; this
    lets Astropy fitters (Astropy 8.1+) fit compound models containing it.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from sunkit_spex.models.instrument_response import MatrixModel
    >>> srm = MatrixModel(np.eye(3), photon_edges=[1, 2, 3, 5] * u.keV, area=2 * u.cm**2, exposure_time=10 * u.s)
    >>> srm(np.ones(3) * u.ph / (u.keV * u.s * u.cm**2))  # doctest: +FLOAT_CMP
    <Quantity [20., 20., 40.] ct>
    """

    name = "SRM"

    # Dimensionless, so a unitless MatrixModel stays fully unitless; the
    # counts-per-photon units belong to the matrix.
    conversion_factor = Parameter(default=1, fixed=True, description="Dimensionless scaling of the response.")

    # Photon bins in, count bins out: the shapes differ by design.
    verify_dims_in_fitting = False
    _input_units_allow_dimensionless = True

    def __init__(
        self,
        matrix=None,
        *,
        photon_edges=None,
        area=None,
        exposure_time=None,
        conversion_factor=1,
        **kwargs,
    ):
        self.matrix = matrix
        self.photon_edges = photon_edges
        self.area = area
        self.exposure_time = exposure_time
        super().__init__(conversion_factor=conversion_factor, **kwargs)

    # ------------------------------------------------------------------ #
    # Validated attributes
    # ------------------------------------------------------------------ #
    @property
    def matrix(self):
        """Response matrix in counts per photon (dimensionless), shape ``(n_photon_bins, n_count_bins)``."""
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if value is None:
            self._matrix = None
            return
        if isinstance(value, Quantity):
            if value.unit.is_equivalent(u.ct / u.ph):
                value = value.to_value(u.ct / u.ph)
            elif value.unit.is_equivalent(u.dimensionless_unscaled):
                value = value.to_value(u.dimensionless_unscaled)
            else:
                raise u.UnitsError(f"Response matrix must be in counts per photon, got {value.unit}.")
        value = np.asarray(value, dtype=float)
        if value.ndim != 2:
            raise ValueError(f"Response matrix must be 2D (n_photon_bins, n_count_bins), got shape {value.shape}.")
        self._matrix = value
        self._check_photon_bins()

    @property
    def photon_edges(self):
        """Photon energy bin edges (keV)."""
        return self._photon_edges

    @photon_edges.setter
    def photon_edges(self, value):
        self._photon_edges = None if value is None else _as_unit(value, u.keV, "photon_edges")
        self._check_photon_bins()

    @property
    def area(self):
        """Area the response is normalised to (cm2)."""
        return self._area

    @area.setter
    def area(self, value):
        self._area = None if value is None else _as_unit(value, u.cm**2, "area")

    @property
    def exposure_time(self):
        """Exposure time (s)."""
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value):
        self._exposure_time = None if value is None else _as_unit(value, u.s, "exposure_time")

    @property
    def is_placeholder(self):
        """`True` if no response matrix has been set yet."""
        return self._matrix is None

    @property
    def n_photon_bins(self):
        """Number of photon (input) bins of the matrix."""
        return None if self.is_placeholder else self._matrix.shape[0]

    @property
    def n_count_bins(self):
        """Number of count (output) bins of the matrix."""
        return None if self.is_placeholder else self._matrix.shape[1]

    def _check_photon_bins(self):
        matrix = getattr(self, "_matrix", None)
        edges = getattr(self, "_photon_edges", None)
        if matrix is not None and edges is not None and edges.size - 1 != matrix.shape[0]:
            raise ValueError(
                f"photon_edges give {edges.size - 1} photon bins but the response matrix has {matrix.shape[0]}."
            )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, flux, conversion_factor):
        if self.is_placeholder:
            raise ValueError(
                "MatrixModel has no response matrix; set one or let FitManager fill it in from the spectrum."
            )

        missing = [name for name in ("photon_edges", "area", "exposure_time") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"MatrixModel needs {', '.join(missing)} to fold a photon flux into counts; set them or let "
                "FitManager fill them in from the spectrum."
            )

        photons_per_flux = np.diff(self._photon_edges) * self._exposure_time * self._area  # keV s cm2

        if isinstance(flux, Quantity) and flux.unit != u.dimensionless_unscaled:
            photons = (flux * photons_per_flux).to_value(u.ph)  # raises unless flux is a photon flux
            # The matrix is in counts per photon.
            return (photons @ self._matrix) * conversion_factor * u.ct

        # Units stripped: plain numbers (as Astropy fitters pass) or dimensionless (as the
        # JointFitter passes for a model without units). Still a photon flux, in
        # ph / (keV s cm2); the output is in ct.
        photons = getattr(flux, "value", flux) * photons_per_flux.to_value(u.keV * u.s * u.cm**2)
        return (photons @ self._matrix) * getattr(conversion_factor, "value", conversion_factor)

    @property
    def input_units(self):
        return {self.inputs[0]: PHOTON_FLUX_UNIT}

    @property
    def return_units(self):
        return {self.outputs[0]: u.ct}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # conversion_factor is dimensionless whatever the data units.
        return {}


def _as_unit(value, unit, name):
    """``value`` as a Quantity in ``unit``; plain numbers are taken to be in ``unit``."""
    if not isinstance(value, Quantity):
        return Quantity(value, unit)
    if not value.unit.is_equivalent(unit):
        raise u.UnitsError(f"{name} must be in units equivalent to {unit}, got {value.unit}.")
    return value.to(unit)