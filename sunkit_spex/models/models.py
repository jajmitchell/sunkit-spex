"""Module for generic mathematical models."""

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

from sunkit_spex.models.count_space import CountSpaceMixin, gaussian_bin_integral
from sunkit_spex.models.instrument_response import PHOTON_FLUX_UNIT

__all__ = ["GaussianModel", "StraightLineModel"]


class StraightLineModel(FittableModel):
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    input_units_equivalencies = {"keV": u.spectral()}

    slope = Parameter(default=1, description="Gradient of a straight line model.")
    intercept = Parameter(default=0, description="Y-intercept of a straight line model.")

    def __init__(self, slope=slope, intercept=intercept, edges=True, **kwargs):
        self.edges = edges

        super().__init__(slope, intercept, **kwargs)

    def evaluate(self, x, slope, intercept):
        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept

    @property
    def input_units(self):
        if isinstance(self.slope, Quantity):
            return {"x": self.intercept.unit / self.slope.unit}
        return None

    @property
    def return_units(self):
        if isinstance(self.slope, Quantity):
            return {"y": self.intercept.unit}
        return None

    def _parameter_units_for_data_units(self, input_units, output_units):
        return {"slope": output_units["y"] / input_units["x"], "intercept": output_units["y"]}


class GaussianModel(CountSpaceMixin, FittableModel):
    r"""
    A Gaussian, :math:`A \exp\left(-(E - E_0)^2 / 2\sigma^2\right)`, in photon or count space.

    The units of the amplitude say which:

    * in photons (e.g. ph s\ :sup:`-1` cm\ :sup:`-2` keV\ :sup:`-1`) it is a
      component of the photon model, folded through the instrument response
      with the rest of it;
    * in counts it describes counts on the spectrum's own (count) energy bins,
      e.g. a calibration-source line, and is added after the response::

          line = GaussianModel(amplitude=5 * u.ct / (u.s * u.keV), mean=31 * u.keV, stddev=1 * u.keV)
          model = ((ThermalEmission() + ThickTarget()) | MatrixModel()) + line

    * without units it is a plain Gaussian, as before.

    Other units are refused. In count space the amplitude as a count flux
    (ct s\ :sup:`-1` keV\ :sup:`-1`, the peak height on a count flux plot) or
    per keV is integrated exactly over each count bin, so a line narrower than
    the bins keeps its counts; per bin (ct s\ :sup:`-1` or ct) it is taken at
    each bin centre. Per unit time it is multiplied by the exposure time.
    `~sunkit_spex.fitting.fitter.FitManager` sets the count bin edges and
    exposure time from each spectrum (see
    `~sunkit_spex.models.count_space.CountSpaceMixin`), and refuses a
    count-space Gaussian folded through a response.

    Parameters
    ----------
    amplitude : `float` or `~astropy.units.Quantity`
        Peak value :math:`A`: in photons for photon space, or in counts (ct,
        ct/s, ct/keV or ct/(s keV)) for count space.
    mean : `float` or `~astropy.units.Quantity`
        Centre :math:`E_0`.
    stddev : `float` or `~astropy.units.Quantity`
        Width :math:`\sigma`; the FWHM is :math:`2 \sqrt{2 \ln 2} \, \sigma
        \approx 2.355 \, \sigma`.
    edges : `bool`, optional
        In photon space, whether the input is bin edges (default), the
        Gaussian being taken at the bin centres, or points.
    count_edges, exposure_time : `~astropy.units.Quantity`, optional
        Count bin edges and exposure time for count space; normally set by
        `~sunkit_spex.fitting.fitter.FitManager`.
    **kwargs
        Passed to `~astropy.modeling.FittableModel`.
    """

    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    _count_parameter = "amplitude"
    _count_parameter_units = {"mean": u.keV, "stddev": u.keV}

    def __init__(
        self,
        amplitude=amplitude,
        mean=mean,
        stddev=stddev,
        edges=True,
        count_edges=None,
        exposure_time=None,
        **kwargs,
    ):
        self.edges = edges

        super().__init__(amplitude, mean, stddev, **kwargs)
        self._set_up_count_space(count_edges, exposure_time)

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""
        if self.count_space:
            return self._evaluate_counts(x, amplitude, mean, stddev)

        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        return amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))

    def _evaluate_counts(self, x, amplitude, mean, stddev):
        """The counts in each count bin."""
        edges = self._count_bin_edges(x)
        (per_time, per_energy), peak = self._count_normalisation(amplitude)
        centre, width = (value.to_value(u.keV) if isinstance(value, Quantity) else value for value in (mean, stddev))
        if per_energy:
            per_bin = peak * gaussian_bin_integral(edges, centre, width)
        else:
            centres = edges[:-1] + 0.5 * np.diff(edges)
            per_bin = peak * np.exp(-0.5 * ((centres - centre) / width) ** 2)
        return self._counts(per_bin, per_time, x, amplitude, mean, stddev)

    @property
    def input_units(self):
        if isinstance(self.mean, Quantity):
            return {"x": self.mean.unit}
        return None

    @property
    def return_units(self):
        if self.count_space:
            return {"y": u.ct}
        if isinstance(self.amplitude, Quantity):
            return {"y": self.amplitude.unit}
        return None

    def _parameter_units_for_data_units(self, input_units, output_units):
        if self.count_space:
            return self._count_space_parameter_units()
        unit = self._check_normalisation_unit()
        # Photons are fitted in the units the models they go through take: the response takes a
        # photon flux, and InverseSquareFluxScaling the emission at the Sun.
        amplitude = next(
            (photons for photons in _PHOTON_UNITS if unit is not None and unit.is_equivalent(photons)),
            output_units["y"],
        )
        return {"mean": input_units["x"], "stddev": input_units["x"], "amplitude": amplitude}


_PHOTON_UNITS = (PHOTON_FLUX_UNIT, u.ph / (u.keV * u.s))