"""
Using models in count space as well as photon space.

Most model components describe a photon flux that is folded through the
instrument response. Some features are better described directly in counts,
on the spectrum's own (count) energy bins, e.g. a calibration-source line or
a detector feature. A model that supports both is in count space when its
normalisation is in counts, and in photon space when it is in photons, and
a count-space component is added after the response::

    line = GaussianModel(amplitude=5 * u.ct / (u.s * u.keV), mean=31 * u.keV, stddev=1 * u.keV)
    model = ((ThermalEmission() + ThickTarget()) | MatrixModel()) + line

`~sunkit_spex.fitting.fitter.FitManager` gives such a component the count
bin edges and exposure time of each spectrum, and refuses one folded through
a response.
"""

import numpy as np
from scipy.special import erf, erfc

import astropy.units as u
from astropy.units import Quantity

__all__ = ["COUNT_FLUX_UNIT", "CountSpaceMixin", "gaussian_bin_integral"]

#: Unit of a count flux.
COUNT_FLUX_UNIT = u.ct / (u.s * u.keV)

# (per time, per energy) -> the unit a count-space normalisation is used in
_COUNT_UNITS = {
    (False, False): u.ct,
    (True, False): u.ct / u.s,
    (False, True): u.ct / u.keV,
    (True, True): COUNT_FLUX_UNIT,
}


class CountSpaceMixin:
    r"""
    Lets a model be used in count space as well as photon space, set by the
    units of its normalisation.

    * In counts (ct, ct s\ :sup:`-1`, ct keV\ :sup:`-1` or ct s\ :sup:`-1`
      keV\ :sup:`-1`) the model is in **count space**: it is evaluated on the
      count energy bins of the spectrum (``count_edges``) whatever it is
      evaluated on as part of a larger model (the photon energy edges when that
      contains a `~sunkit_spex.models.instrument_response.MatrixModel`), and
      returns the counts in each count bin, so it is added after the response.
      A normalisation per keV is integrated over each bin, one per unit time
      is multiplied by the exposure time (``exposure_time``), and one per bin
      is used at each bin.
    * In photons (any units with ph) the model is in **photon space**, folded
      through the response with the rest of the photon model.
    * Without units it is a plain mathematical model, as in photon space.

    Any other units are refused. ``count_space`` says which space the model is
    in; to move it, give the normalisation new units.

    `~sunkit_spex.fitting.fitter.FitManager` sets ``count_edges`` (the bins
    being fitted, or every bin when plotting the full spectrum) and
    ``exposure_time`` from each spectrum. Without ``count_edges`` the model is
    evaluated on its input, taken as count bin edges.

    A model using this names its normalisation in ``_count_parameter`` and
    the units its other parameters are fitted in, in count space, in
    ``_count_parameter_units``; calls ``_set_up_count_space`` at the end of its
    ``__init__``; and, in ``evaluate``, uses ``_count_bin_edges``,
    ``_count_normalisation`` and ``_counts`` when ``count_space`` is `True`.
    """

    #: The parameter whose units set the space, and how the model becomes counts.
    _count_parameter = None
    #: Units the other parameters are fitted in when in count space.
    _count_parameter_units = {}

    def _set_up_count_space(self, count_edges, exposure_time):
        self._normalisation_unit = None
        self.count_edges = count_edges
        self.exposure_time = exposure_time
        self._check_normalisation_unit()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == self._count_parameter and "_normalisation_unit" in self.__dict__:
            self._check_normalisation_unit()

    def _check_normalisation_unit(self):
        """
        The normalisation's unit, checked to be counts or photons. Remembered,
        as Astropy fitters strip the units and later ask for them again.
        """
        unit = getattr(self, self._count_parameter).unit
        if unit is not None and unit != u.dimensionless_unscaled and unit != self._normalisation_unit:
            _check_space(unit, type(self).__name__)
            self._normalisation_unit = unit
        return self._normalisation_unit

    @property
    def count_space(self):
        """`True` in count space (normalisation in counts), `False` in photon space; set by the units."""
        return _count_form(self._check_normalisation_unit()) is not None

    @property
    def count_edges(self):
        """Count energy bin edges (keV) the model is evaluated on in count space."""
        return self._count_edges

    @count_edges.setter
    def count_edges(self, value):
        self._count_edges = None if value is None else _as_unit(value, u.keV, "count_edges")

    @property
    def exposure_time(self):
        """Exposure time (s), for a normalisation per unit time in count space."""
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value):
        self._exposure_time = None if value is None else _as_unit(value, u.s, "exposure_time")

    @property
    def needs_exposure_time(self):
        """`True` if the model needs an exposure time to give counts (a normalisation per unit time)."""
        form = _count_form(self._check_normalisation_unit())
        return form is not None and form[0]

    def _count_bin_edges(self, x):
        """The count bin edges in keV, as plain numbers."""
        edges = self._count_edges if self._count_edges is not None else x
        return edges.to_value(u.keV) if isinstance(edges, Quantity) else np.asarray(edges, dtype=float)

    def _count_normalisation(self, value):
        """``(per_time, per_energy)`` and the normalisation's value in its count unit."""
        unit = value.unit if isinstance(value, Quantity) and value.unit != u.dimensionless_unscaled else None
        if unit is not None:
            form = _count_form(unit)
            return form, value.to_value(_COUNT_UNITS[form])
        # Units stripped (as Astropy fitters do): in the count unit for its form.
        return _count_form(self._normalisation_unit), getattr(value, "value", value)

    def _counts(self, per_bin, per_time, *inputs):
        """
        Counts from ``per_bin`` (the normalisation integrated over or taken in each
        bin), with units if any of ``inputs`` has them.
        """
        if per_time:
            if self._exposure_time is None:
                raise ValueError(
                    f"{type(self).__name__} needs an exposure_time to give counts from a normalisation per unit "
                    "time; set it or let FitManager fill it in from the spectrum."
                )
            per_bin = per_bin * self._exposure_time.to_value(u.s)
        # With units when evaluated with units (as the joint fitters do, calling evaluate directly);
        # plain numbers when the units have been stripped (as Astropy fitters do).
        if any(isinstance(value, Quantity) and value.unit != u.dimensionless_unscaled for value in inputs):
            return per_bin * u.ct
        return per_bin

    def _count_space_parameter_units(self):
        """Units the parameters are fitted in when in count space, whatever the data units."""
        form = _count_form(self._check_normalisation_unit())
        return {self._count_parameter: _COUNT_UNITS[form]} | dict(self._count_parameter_units)


def gaussian_bin_integral(edges, mean, stddev):
    r"""
    The integral of :math:`\exp(-(E - E_0)^2 / 2\sigma^2)` over each bin.

    Exact, and precise in the tails, so narrow lines keep their area however
    they fall on the bins.

    Parameters
    ----------
    edges : `numpy.ndarray`
        Bin edges.
    mean, stddev : `float`
        :math:`E_0` and :math:`\sigma`, in the units of ``edges``.

    Returns
    -------
    `numpy.ndarray`
        One value per bin, in the units of ``edges``.
    """
    stddev = np.maximum(np.abs(stddev), np.finfo(float).tiny)
    z = (edges - mean) / (np.sqrt(2) * stddev)
    low, high = z[:-1], z[1:]
    # erf(high) - erf(low), through erfc where both are close to 1 or -1
    difference = np.where(
        low > 0,
        erfc(low) - erfc(high),
        np.where(high < 0, erfc(-high) - erfc(-low), erf(high) - erf(low)),
    )
    return stddev * np.sqrt(np.pi / 2) * difference


def _count_form(unit):
    """``(per_time, per_energy)`` for a normalisation in counts; `None` in photons or without units."""
    if unit is None:
        return None
    for form, count_unit in _COUNT_UNITS.items():
        if unit.is_equivalent(count_unit):
            return form
    return None


def _check_space(unit, owner):
    """Raise unless ``unit`` is counts (in one of the count forms) or photons."""
    if _count_form(unit) is not None:
        return
    bases = unit.decompose().bases
    if u.ph in bases and u.ct not in bases:
        return
    raise u.UnitsError(
        f"The {owner} normalisation must be in photons, for photon space (folded through the response), or in "
        f"counts, for count space (added after the response): ct, ct/s, ct/keV or ct/(s keV); got {unit}."
    )


def _as_unit(value, unit, name):
    """``value`` as a Quantity in ``unit``; plain numbers are taken to be in ``unit``."""
    if not isinstance(value, Quantity):
        return Quantity(value, unit)
    if not value.unit.is_equivalent(unit):
        raise u.UnitsError(f"{name} must be in units equivalent to {unit}, got {value.unit}.")
    return value.to(unit)
