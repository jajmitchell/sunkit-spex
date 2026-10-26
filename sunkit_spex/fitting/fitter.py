"""
Manage fitting of Astropy models to one or more `~sunkit_spex.spectrum.Spectrum` objects.

`FitManager` takes care of the bookkeeping between a spectrum and a model:
it selects the fit range, fills in the instrument-specific pieces of the
model (response matrix, albedo energy grid and angle, observer distance)
from the spectrum and its metadata, and hands everything to a fitter.

The model components are found by walking the model's leaf list (the
ordered list of simple models a compound model is built from) and
dispatching on their type, so they do not need any particular name.
"""

import copy
import warnings
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.modeling import CompoundModel, Model, fitting
from astropy.nddata import StdDevUncertainty

from sunkit_spex.fitting.fitters import JointFitter, ScipyMinimizeJointFitter
from sunkit_spex.models.instrument_response import MatrixModel
from sunkit_spex.models.physical.albedo import Albedo
from sunkit_spex.models.scaling import InverseSquareFluxScaling
from sunkit_spex.spectrum import Spectrum

__all__ = ["DEFAULT_META_KEYS", "FitManager", "leaf_models"]


#: Default mapping from the quantities `FitManager` needs to the keys of
#: ``Spectrum.meta`` they are read from. Override per instance with the
#: ``meta_keys`` argument.
DEFAULT_META_KEYS = {
    "srm": "srm",
    "photon_edges": "ph_axis",
    "area": "geo_area",
    "exposure_time": "exposure_time",
    "albedo_angle": "angle",
    "observer_distance": "distance",
}

# Warnings point at the user's code rather than at this module.
_WARN_SKIP = (str(Path(__file__).parent),)

# Units assumed for metadata values given as plain numbers.
# (MatrixModel checks and converts the units of the values it is given.)
_DEFAULT_META_UNITS = {
    "area": u.cm**2,
    "exposure_time": u.s,
    "albedo_angle": u.deg,
    "observer_distance": u.AU,
}


def leaf_models(model):
    """
    Return the simple models a model is built from, in evaluation order.

    For a `~astropy.modeling.CompoundModel` this is its leaf list, whose
    order matches the suffix Astropy gives each parameter (the leaf at
    index ``i`` owns the parameters ending in ``_i``). A simple model is
    its own only leaf.

    The leaves share their `~astropy.modeling.Parameter` objects with the
    compound model, so changing a leaf's parameter value, bounds or fixed
    state changes the compound model too.

    Parameters
    ----------
    model : `~astropy.modeling.Model`

    Returns
    -------
    `list` [`~astropy.modeling.Model`]
    """
    if isinstance(model, CompoundModel):
        if model._leaflist is None:
            model._make_leaflist()
        return list(model._leaflist)
    return [model]


class FitManager:
    r"""
    Fit Astropy models to one or more spectra, optionally jointly.

    Parameters
    ----------
    models : `~astropy.modeling.Model` or `list` [`~astropy.modeling.Model`]
        Model, or one model per spectrum. The models are copied, so the
        originals are never modified.
    spectra : `~sunkit_spex.spectrum.Spectrum` or `list` [`~sunkit_spex.spectrum.Spectrum`]
        Spectrum, or list of spectra, to fit.
    fitter : `~sunkit_spex.fitting.fitters.JointFitter` or Astropy fitter, optional
        Fitter to use. A joint fitter fits all model/spectrum pairs
        simultaneously; a standard Astropy fitter fits a single pair, see
        Notes. Defaults to ``TRFLSQFitter(calc_uncertainties=True)`` for a
        single spectrum and to
        `~sunkit_spex.fitting.fitters.ScipyMinimizeJointFitter` for several.
    fit_range : `tuple` or `list` [`tuple` or `None`], optional
        ``(min, max)`` energy range to fit, applied to every spectrum, or one
        range (or `None` for all bins) per spectrum. Only bins lying entirely
        within the range are fitted. Plain numbers are taken to be in the
        units of the spectral axis.
    meta_keys : `dict`, optional
        Overrides for `DEFAULT_META_KEYS`, mapping what is needed to the
        ``Spectrum.meta`` key it is stored under.

    Notes
    -----
    The model's leaves are configured from each spectrum as follows:

    * `~sunkit_spex.models.instrument_response.MatrixModel` created without
      a matrix is filled with the response in ``meta["srm"]`` (shape
      ``(n_photon_bins, n_count_bins)``), restricted to the fitted count bins,
      and the photon energy edges (``meta["ph_axis"]``), area
      (``meta["geo_area"]``) and exposure time (``meta["exposure_time"]``).
      The model is then evaluated on the photon energy edges.
    * `~sunkit_spex.models.physical.albedo.Albedo` gets the photon energy
      edges and, if present in the metadata, the observer angle (fixed). If
      the angle is missing a warning is raised and ``theta`` keeps the value
      (and fixed state) it has in the model.
    * `~sunkit_spex.models.scaling.InverseSquareFluxScaling` gets the
      observer distance (fixed), if present in the metadata.

    All fitting is done in count space. Spectra given as a count rate
    (ct s\ :sup:`-1`), per unit energy (ct keV\ :sup:`-1`) or both
    (ct s\ :sup:`-1` keV\ :sup:`-1`), in any equivalent units, are converted
    to counts first by multiplying the data and uncertainty by the exposure
    time (``meta["exposure_time"]``) and/or the count bin widths. Each
    prepared model is then evaluated once to check that it returns counts
    for every fitted bin, so a photon-space model needs a
    `~sunkit_spex.models.instrument_response.MatrixModel` to fold it into
    counts.

    When fitting several spectra, parameters can be tied across models using
    the model list passed to the joint fitter, e.g.
    ``model_b.temperature_0.tied = lambda models: models[0].temperature_0``.

    Unless given explicitly, fit weights are derived from the spectrum
    uncertainty ``sigma`` following each fitter's convention: ``1/sigma**2``
    for joint fitters and ``1/sigma`` for Astropy fitters.

    Spectral models are evaluated on bin edges (or on a photon energy grid
    when a response is used), so ``x`` and ``y`` have different shapes.
    Astropy fitters accept this when a component of the model sets
    ``verify_dims_in_fitting = False`` (Astropy 8.1+), as
    `~sunkit_spex.models.instrument_response.MatrixModel` and the edge-based
    spectral models do.
    Unit-aware fitting with an Astropy fitter also needs every component to
    give its parameter units independently of the data units, as the
    sunkit-spex physical models do.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from sunkit_spex.fitting.fit_manager import FitManager
    >>> from sunkit_spex.models.models import StraightLineModel
    >>> from sunkit_spex.spectrum import Spectrum
    >>> edges = np.arange(1, 12) * u.keV
    >>> spec = Spectrum((2 * np.arange(1.5, 11.5) + 1) * u.ct, spectral_axis=edges)
    >>> model = StraightLineModel(slope=1 * u.ct / u.keV, intercept=0 * u.ct)
    >>> fm = FitManager(model, spec, fit_range=(2, 9) * u.keV)
    >>> fitted = fm.fit()
    >>> fitted.parameters  # doctest: +FLOAT_CMP
    array([2., 1.])
    """

    def __init__(self, models, spectra, fitter=None, fit_range=None, meta_keys=None):
        models = [models] if isinstance(models, Model) else list(models)
        spectra = list(spectra) if isinstance(spectra, (list, tuple)) else [spectra]
        if len(models) != len(spectra):
            raise ValueError(f"Got {len(models)} model(s) but {len(spectra)} spectra; need one model per spectrum.")
        if len(models) == 0:
            raise ValueError("At least one model and spectrum are required.")

        self._models = models
        self._spectra = spectra
        self._meta_keys = DEFAULT_META_KEYS | (meta_keys or {})
        self.fitter = self._default_fitter() if fitter is None else fitter
        self.fit_range = fit_range

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def models(self):
        """The models as given (never modified)."""
        return list(self._models)

    @property
    def spectra(self):
        """The spectra being fitted."""
        return list(self._spectra)

    @property
    def is_joint(self):
        """`True` if more than one spectrum is being fitted."""
        return len(self._spectra) > 1

    @property
    def fitter(self):
        """The fitter used by `~FitManager.fit`."""
        return self._fitter

    @fitter.setter
    def fitter(self, value):
        if self.is_joint and not isinstance(value, JointFitter):
            raise TypeError(
                f"Fitting {len(self._spectra)} spectra together needs a JointFitter, got {type(value).__name__}."
            )
        self._fitter = value
        self._reset()

    def _default_fitter(self):
        if self.is_joint:
            return ScipyMinimizeJointFitter()
        return fitting.TRFLSQFitter(calc_uncertainties=True)

    @property
    def _uses_joint_fitter(self):
        return isinstance(self._fitter, JointFitter)

    @property
    def fit_range(self):
        """Energy range fitted for each spectrum (`list` of ``(min, max)`` or `None`)."""
        return list(self._fit_range)

    @fit_range.setter
    def fit_range(self, value):
        if value is None or _is_single_range(value):
            ranges = [value] * len(self._spectra)
        else:
            ranges = list(value)
            if len(ranges) != len(self._spectra):
                raise ValueError(f"Got {len(ranges)} fit ranges for {len(self._spectra)} spectra.")
        self._fit_range = [None if r is None else _validate_range(r) for r in ranges]
        self._reset()

    @property
    def prepared_models(self):
        """
        `list` [`~astropy.modeling.Model`]: copies of the models with their
        spectrum-specific components (response, albedo, distance) filled in,
        before fitting.
        """
        return [model for model, _ in self._get_prepared()]

    @property
    def fit_spectra(self):
        """
        `list` [`~sunkit_spex.spectrum.Spectrum`]: the spectra as fitted: cut
        to their fit range (with bin edges and response matrix cut to match)
        and converted to counts.
        """
        return [spectrum for _, spectrum in self._get_prepared()]

    @property
    def fitted_models(self):
        """`list` [`~astropy.modeling.Model`]: fitted model per spectrum, available after `~FitManager.fit`."""
        if self._fitted_models is None:
            raise RuntimeError("No fitted models available, call fit() first.")
        return list(self._fitted_models)

    @property
    def fitted_model(self):
        """The fitted model when fitting a single spectrum."""
        if self.is_joint:
            raise AttributeError("Several spectra were fitted, use fitted_models.")
        return self.fitted_models[0]

    @property
    def fit_info(self):
        """Fit information recorded by the fitter."""
        return getattr(self._fitter, "fit_info", None)

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, **kwargs):
        """
        Fit the model(s) to the spectrum (spectra).

        Parameters
        ----------
        **kwargs :
            Passed to the fitter. For a joint fitter these are ``fkwarg``
            (objective function keywords, where ``"weights"`` is filled in
            from the spectrum uncertainties unless given) and ``optkwarg``
            (optimizer keywords). For an Astropy fitter these are its usual
            keywords (e.g. ``maxiter``); ``weights`` is filled in from the
            spectrum uncertainty unless given.

        Returns
        -------
        `~astropy.modeling.Model` or `list` [`~astropy.modeling.Model`]
            The fitted model, or list of fitted models when fitting jointly.
        """
        prepared = self._get_prepared()
        sigmas = [_stddev(spectrum) for _, spectrum in prepared]

        if self._uses_joint_fitter:
            fkwarg = dict(kwargs.pop("fkwarg", None) or {})
            if "weights" not in fkwarg and all(sigma is not None for sigma in sigmas):
                fkwarg["weights"] = [_weights(sigma, power=2) for sigma in sigmas]
            args = [
                item
                for model, spectrum in prepared
                for item in (model, self._model_x(model, spectrum), _data(spectrum))
            ]
            # Fit copies so the prepared models keep their initial values.
            fitted = self._fitter(*args, fkwarg=fkwarg, inplace=False, **kwargs)
        else:
            ((model, spectrum),) = prepared
            (sigma,) = sigmas
            if "weights" not in kwargs and sigma is not None:
                kwargs["weights"] = _weights(sigma, power=1)
            # Astropy fitters always return a fitted copy.
            fitted = [self._fitter(model, self._model_x(model, spectrum), _data(spectrum), **kwargs)]

        self._fitted_models = list(fitted)
        return self._fitted_models if self.is_joint else self._fitted_models[0]

    def evaluate(self, fitted=True):
        """
        Evaluate the models on their fitted energy grid.

        Parameters
        ----------
        fitted : `bool`, optional
            Use the fitted models (default) or the initial ones.

        Returns
        -------
        `list` [`~astropy.units.Quantity`]
            Model values in data space, one array per spectrum.
        """
        prepared = self._get_prepared()
        models = self.fitted_models if fitted else [model for model, _ in prepared]
        return [model(self._model_x(model, spectrum)) for model, (_, spectrum) in zip(models, prepared)]

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #
    def _reset(self):
        self._prepared = None
        self._fitted_models = None

    def _get_prepared(self):
        """``(model, spectrum)`` per spectrum, ready to fit; built on first use."""
        if self._prepared is None:
            self._prepared = [
                self._prepare(model, spectrum, fit_range)
                for model, spectrum, fit_range in zip(self._models, self._spectra, self._fit_range)
            ]
        return self._prepared

    def _model_x(self, model, spectrum):
        """Energy edges ``model`` is evaluated on for ``spectrum``.

        Photon edges when the model folds through a response, otherwise the
        spectrum's own (count) bin edges.
        """
        matrices = [leaf for leaf in leaf_models(model) if isinstance(leaf, MatrixModel)]
        if matrices:
            if matrices[0].photon_edges is not None:
                return matrices[0].photon_edges
            return self._photon_edges(spectrum, required_for="evaluating a model with a MatrixModel")
        return u.Quantity(spectrum.spectral_axis.bin_edges)

    def _meta(self, spectrum, name, required_for=None):
        key = self._meta_keys[name]
        if key in spectrum.meta:
            value = spectrum.meta[key]
            if name in _DEFAULT_META_UNITS and not isinstance(value, u.Quantity):
                value = u.Quantity(value, _DEFAULT_META_UNITS[name])
            return value
        if required_for is not None:
            raise KeyError(f"Spectrum meta has no {key!r} ({name}), needed for {required_for}.")
        return None

    def _photon_edges(self, spectrum, required_for):
        edges = self._meta(spectrum, "photon_edges", required_for=required_for)
        return edges if isinstance(edges, u.Quantity) else u.Quantity(edges, spectrum.spectral_axis.unit)

    def _prepare(self, model, spectrum, fit_range):
        """Cut ``spectrum`` to ``fit_range`` and fill ``model``'s components from it."""
        if spectrum.data.ndim != 1:
            raise ValueError(f"FitManager only supports 1D spectra, got data with shape {spectrum.data.shape}.")
        if spectrum.spectral_axis.bin_edges is None:
            raise ValueError("Spectrum spectral axis must be defined by bin edges.")

        bins = _bins_in_range(u.Quantity(spectrum.spectral_axis.bin_edges), fit_range)
        spectrum = _slice_spectrum(spectrum, bins, srm_key=self._meta_keys["srm"])
        spectrum = self._to_counts(spectrum)

        model = copy.deepcopy(model)
        for leaf in leaf_models(model):
            if isinstance(leaf, MatrixModel):
                if leaf.is_placeholder:
                    self._configure_matrix(leaf, spectrum)
                if leaf.n_count_bins != spectrum.data.size:
                    raise ValueError(
                        f"MatrixModel has {leaf.n_count_bins} count bins but {spectrum.data.size} bins are being "
                        "fitted."
                    )
            elif isinstance(leaf, Albedo):
                leaf.energy_edges = self._photon_edges(spectrum, required_for="the albedo correction")
                angle = self._meta(spectrum, "albedo_angle")
                if angle is not None:
                    leaf.theta = angle
                    leaf.theta.fixed = True
                else:
                    state = "fixed" if leaf.theta.fixed else "free"
                    warnings.warn(
                        f"Spectrum meta has no {self._meta_keys['albedo_angle']!r} (albedo angle); the Albedo "
                        f"theta keeps its model value of {leaf.theta.quantity} and is {state}.",
                        skip_file_prefixes=_WARN_SKIP,
                    )
            elif isinstance(leaf, InverseSquareFluxScaling):
                distance = self._meta(spectrum, "observer_distance")
                if distance is not None:
                    leaf.observer_distance = distance
                    leaf.observer_distance.fixed = True

        self._check_count_space(model, spectrum)
        return model, spectrum

    def _to_counts(self, spectrum):
        """
        The spectrum in counts: data and uncertainty multiplied by the exposure
        time and/or count bin widths when given as a rate or per unit energy.
        """
        if spectrum.unit is None:
            raise u.UnitsError("FitManager fits in count space, so the spectrum data must have units.")
        if spectrum.unit.is_equivalent(u.ct):
            if spectrum.unit == u.ct:
                return spectrum
            per_time = per_energy = False
        else:
            per_time, per_energy = _count_form(spectrum.unit, spectrum.spectral_axis.unit)

        factor = 1 * u.dimensionless_unscaled
        if per_time:
            exposure = self._meta(
                spectrum, "exposure_time", required_for=f"converting data in {spectrum.unit} to counts"
            )
            factor = factor * exposure
        if per_energy:
            factor = factor * np.diff(u.Quantity(spectrum.spectral_axis.bin_edges))

        counts = (u.Quantity(spectrum.data, spectrum.unit) * factor).to(u.ct)
        sigma = _stddev(spectrum)
        uncertainty = None if sigma is None else StdDevUncertainty((sigma * factor).to_value(u.ct), unit=u.ct)

        return Spectrum(
            counts,
            spectral_axis=u.Quantity(spectrum.spectral_axis.bin_edges),
            uncertainty=uncertainty,
            meta=spectrum.meta,
        )

    def _check_count_space(self, model, spectrum):
        """Check the prepared model gives counts for every fitted bin."""
        output = model(self._model_x(model, spectrum))
        if not isinstance(output, u.Quantity) or not output.unit.is_equivalent(u.ct):
            unit = getattr(output, "unit", None)
            raise u.UnitsError(
                f"The model must return counts to be fitted in count space, got {unit}. Fold a photon-space "
                "model through a MatrixModel, and give the model parameters units."
            )
        if output.shape != spectrum.data.shape:
            raise ValueError(f"The model returns {output.shape} values for {spectrum.data.shape} fitted bins.")

    def _configure_matrix(self, leaf, spectrum):
        """Fill a placeholder MatrixModel from the (fit-range cut) spectrum's metadata."""
        needed = "the MatrixModel"
        # MatrixModel checks the units and that the matrix matches the photon edges.
        leaf.matrix = self._meta(spectrum, "srm", required_for=needed)
        leaf.photon_edges = self._photon_edges(spectrum, required_for=needed)
        leaf.area = self._meta(spectrum, "area", required_for=needed)
        leaf.exposure_time = self._meta(spectrum, "exposure_time", required_for=needed)


def _count_form(data_unit, energy_unit):
    """``(per_time, per_energy)`` for count-like data units, e.g. ``(True, True)`` for ct/(s keV)."""
    for per_time, per_energy in ((True, False), (False, True), (True, True)):
        unit = u.ct / (u.s if per_time else 1) / (energy_unit if per_energy else 1)
        if data_unit.is_equivalent(unit):
            return per_time, per_energy
    raise u.UnitsError(
        f"FitManager fits in count space; can not convert data in {data_unit} to counts. Expected counts, a count "
        "rate, counts per unit energy, or a count rate per unit energy."
    )


def _is_single_range(value):
    """`True` for one ``(min, max)`` range rather than a list of ranges."""
    if isinstance(value, u.Quantity):
        return value.shape == (2,)
    return len(value) == 2 and all(np.isscalar(v) or (isinstance(v, u.Quantity) and v.isscalar) for v in value)


def _validate_range(fit_range):
    lower, upper = fit_range
    if lower >= upper:
        raise ValueError(f"Fit range lower bound must be below the upper bound, got {fit_range}.")
    return fit_range


def _bins_in_range(edges, fit_range):
    """Slice of the bins lying entirely within ``fit_range``."""
    if fit_range is None:
        return slice(0, edges.size - 1)
    lower, upper = (u.Quantity(v, edges.unit) for v in fit_range)
    inside = np.flatnonzero((edges[:-1] >= lower) & (edges[1:] <= upper))
    if inside.size == 0:
        raise ValueError(f"No bins lie entirely within the fit range {fit_range}.")
    return slice(inside[0], inside[-1] + 1)


def _slice_spectrum(spectrum, bins, srm_key):
    """
    ``spectrum[bins]`` with its bin edges and response matrix cut to match.

    Slicing a `~sunkit_spex.spectrum.Spectrum` cuts the data, uncertainty and
    bin centres, but not yet the ``SpectralAxis`` bin edges or anything in
    ``meta``. Until that is fixed in the spectrum module, set them here.
    """
    edges = u.Quantity(spectrum.spectral_axis.bin_edges)
    srm = spectrum.meta.get(srm_key)
    if srm is not None:
        srm = np.asanyarray(srm)
    if srm is not None and (srm.ndim != 2 or srm.shape[1] != edges.size - 1):
        raise ValueError(
            f"Response matrix in meta[{srm_key!r}] has shape {srm.shape}; expected (n_photon_bins, "
            f"{edges.size - 1}) for this spectrum."
        )

    sliced = spectrum[bins]

    axis = sliced.spectral_axis
    axis._bin_edges = edges[bins.start : bins.stop + 1]
    axis.__dict__.pop("bin_edges", None)  # drop any cached value of the lazy property

    # The sliced spectrum shares ``meta`` with the original, so give it its own.
    meta = dict(spectrum.meta)
    if srm is not None:
        meta[srm_key] = srm[:, bins]
    sliced.meta = meta
    return sliced


def _data(spectrum):
    """The spectrum's data as a Quantity."""
    return u.Quantity(spectrum.data, spectrum.unit)


def _stddev(spectrum):
    """The spectrum's 1-sigma uncertainty as a Quantity, or None."""
    uncertainty = spectrum.uncertainty
    if uncertainty is None:
        return None
    unit = uncertainty.unit or spectrum.unit
    kind = uncertainty.uncertainty_type
    if kind in ("std", "poisson"):
        return u.Quantity(uncertainty.array, unit)
    if kind == "var":
        return np.sqrt(u.Quantity(uncertainty.array, unit))
    if kind == "ivar":
        return 1 / np.sqrt(u.Quantity(uncertainty.array, unit))
    raise TypeError(f"Unsupported uncertainty type {kind!r}.")


def _weights(stddev, power):
    """``1/stddev**power``, with zero weight where the uncertainty is not usable."""
    bad = ~np.isfinite(stddev) | (stddev <= 0)
    if np.any(bad):
        warnings.warn(
            f"{np.count_nonzero(bad)} bin(s) have zero or invalid uncertainty and are given zero weight.",
            skip_file_prefixes=_WARN_SKIP,
        )
    safe = np.where(bad, 1 * stddev.unit, stddev)
    return np.where(bad, 0, 1 / safe**power)