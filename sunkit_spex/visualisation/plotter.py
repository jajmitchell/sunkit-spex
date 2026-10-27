"""
Plot the results of fitting with `~sunkit_spex.fitting.fitter.FitManager`.

The plotter works with any model built from Astropy models. The fitted model
is split into its additive components by walking its expression tree:

* ``A + B`` gives the components of ``A`` and of ``B``;
* ``A * B`` multiplies out, so ``(thermal + thick) * distance`` gives a
  thermal and a thick-target component, each scaled by the distance;
* ``A | R``, where ``R`` is linear (a response matrix or albedo), passes each
  component of ``A`` through ``R`` separately. An albedo correction is split
  into the direct component and a reflected one, all reflected parts being
  shown together as "Albedo".

Each component is named after the model it comes from, ignoring pure
scalings (distance, constants) and the response. The components always add
up to the fitted model.
"""

import copy
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like, to_rgb, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from matplotlib.transforms import blended_transform_factory, offset_copy

import astropy.units as u
from astropy.modeling import CompoundModel, Parameter
from astropy.time import Time

from sunkit_spex.models.instrument_response import MatrixModel
from sunkit_spex.models.physical.albedo import Albedo
from sunkit_spex.models.scaling import Constant, InverseSquareFluxScaling

__all__ = ["COMPONENT_COLOURS", "DISPLAY_NAMES", "PARAMETER_LABELS", "RC_PARAMS", "model_components", "plot_fit"]

_WARN_SKIP = (str(Path(__file__).parent),)

# Shades of the usual red, blue and green, for several components of one kind.
_THERMAL_COLOURS = ("#d62728", "#8c1a1a", "#f08a87", "#b5446e")
_NONTHERMAL_COLOURS = ("#1f77b4", "#0b3d6e", "#6baed6", "#5e4fa2")
_ALBEDO_COLOURS = ("#2ca02c", "#17631b", "#8fd18a", "#7a9a1a")

#: Colours of each kind of component, by model class name: a single colour,
#: or a sequence used in turn when a model has several components of the kind
#: (two thermal components, say). Kinds given the same sequence share it, so a
#: thick- and a thin-target component get different blues. Other kinds take
#: the next free colour from ``_SPARE_COLOURS``, and shades of it.
COMPONENT_COLOURS = {
    "ThermalEmission": _THERMAL_COLOURS,
    "ContinuumEmission": _THERMAL_COLOURS,
    "LineEmission": _THERMAL_COLOURS,
    "ThickTarget": _NONTHERMAL_COLOURS,
    "ThinTarget": _NONTHERMAL_COLOURS,
    "Albedo": _ALBEDO_COLOURS,
}
_SPARE_COLOURS = ["C1", "C4", "C5", "C6", "C8", "C9", "C7"]
TOTAL_COLOUR = "k"

#: Names components are shown with, by model name. Models named by the user
#: (``name=...``) are shown with that name.
DISPLAY_NAMES = {
    "ThermalEmission": "Thermal",
    "ContinuumEmission": "Thermal continuum",
    "LineEmission": "Thermal lines",
    "ThickTarget": "Thick target",
    "ThinTarget": "Thin target",
    "Albedo": "Albedo",
    "InverseSquareFluxScaling": "Distance",
    "Constant": "Constant",
}

#: Display symbol and, where the model works in scaled units, the unit string
#: for each parameter, by ``(model class name, parameter name)`` or parameter
#: name alone. Anything else is shown by name with its own unit.
PARAMETER_LABELS = {
    "temperature": ("T", None),
    ("ThermalEmission", "emission_measure"): ("EM", r"$\times 10^{49}\,\mathrm{cm^{-3}}$"),
    ("ContinuumEmission", "emission_measure"): ("EM", r"$\times 10^{49}\,\mathrm{cm^{-3}}$"),
    ("LineEmission", "emission_measure"): ("EM", r"$\times 10^{49}\,\mathrm{cm^{-3}}$"),
    "p": (r"\delta", None),
    "q": (r"\delta_{2}", None),
    "break_energy": (r"E_{\mathrm{break}}", None),
    "low_e_cutoff": (r"E_{\mathrm{c}}", None),
    "high_e_cutoff": (r"E_{\mathrm{high}}", None),
    ("ThickTarget", "total_eflux"): (r"F_{e}", r"$\times 10^{35}\,\mathrm{e^{-}\,s^{-1}}$"),
    ("ThinTarget", "total_eflux"): (r"F_{e}", r"$\times 10^{55}\,\mathrm{cm^{-2}\,s^{-1}}$"),
    "theta": (r"\theta", None),
    "anisotropy": (r"a", None),
    "observer_distance": (r"D", None),
    "constant": (r"C", None),
    "conversion_factor": (r"c", None),
}

# Models that only scale or fold a component; they don't name it.
_MODIFIERS = (InverseSquareFluxScaling, Constant, MatrixModel)
# The reflected albedo parts of every component are collected as one.
_REFLECTED = ("Albedo", False)
_JOIN = " \N{MULTIPLICATION SIGN} "

#: matplotlib settings for the plots: fully enclosed axes with outward ticks
#: and a faint grid. Applied only while a plot is drawn.
RC_PARAMS = {
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.15",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "0.15",
    "ytick.color": "0.15",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "legend.handlelength": 1.6,
    "legend.labelspacing": 0.25,
}
_FIGSIZE = (8, 6)
# Size of each panel in a grid, in inches.
_GRID_CELL = (2.9, 2.5)
# Height of the gap between rows of panels, relative to a spectrum (3) and its residuals (1).
_GRID_GAP = 0.55
# Smaller text for grids of many small panels.
_SMALL_PANEL_RC = {
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 9,
}
_TOTAL_WIDTH = 1.8
_COMPONENT_WIDTH = 1.6

_Y_UNITS = {
    "flux": (True, True, r"Count flux ($\mathrm{ct\,s^{-1}\,keV^{-1}}$)"),
    "rate": (True, False, r"Count rate ($\mathrm{ct\,s^{-1}}$)"),
    "counts": (False, False, "Counts"),
}


# ---------------------------------------------------------------------- #
# Splitting a model into components
# ---------------------------------------------------------------------- #
def _leaves(model):
    if not isinstance(model, CompoundModel):
        return [model]
    if model._leaflist is None:
        model._make_leaflist()
    return list(model._leaflist)


def _leaf_labels(model, names=None):
    """``{id(leaf): (display label, is_modifier)}``, numbering repeated names."""
    names = names or {}
    leaves = _leaves(model)
    base = []
    for leaf in leaves:
        name = leaf.name or type(leaf).__name__
        base.append(names.get(name) or DISPLAY_NAMES.get(name, name))
    labels, seen = {}, {}
    for leaf, name in zip(leaves, base):
        if base.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name} {seen[name]}"
        labels[id(leaf)] = (name, isinstance(leaf, _MODIFIERS))
    return labels


def _merge(key_a, key_b):
    if _REFLECTED in key_a or _REFLECTED in key_b:
        return (_REFLECTED,)
    return key_a + tuple(k for k in key_b if k not in key_a)


def _terms(node, inp, labels):
    """``[(key, value), ...]`` whose values add up to ``node(inp)``."""
    if not isinstance(node, CompoundModel):
        if isinstance(node, Albedo):
            # Linear in its input: direct + reflected.
            return [((), inp), ((_REFLECTED,), node(inp) - inp)]
        return [((labels[id(node)],), node(inp))]

    op = node.op
    if op == "+":
        return _terms(node.left, inp, labels) + _terms(node.right, inp, labels)
    if op == "-":
        return _terms(node.left, inp, labels) + [(k, -v) for k, v in _terms(node.right, inp, labels)]
    if op == "*":
        left, right = _terms(node.left, inp, labels), _terms(node.right, inp, labels)
        return [(_merge(kl, kr), vl * vr) for kl, vl in left for kr, vr in right]
    if op == "/":
        denominator = node.right(inp)
        return [(k, v / denominator) for k, v in _terms(node.left, inp, labels)]
    if op == "|" and _is_linear(node.right):
        return [
            (_merge(kl, kr), vr)
            for kl, vl in _terms(node.left, inp, labels)
            for kr, vr in _terms(node.right, vl, labels)
        ]
    # Anything else can't be split: keep it as one component.
    key = tuple(labels[id(leaf)] for leaf in node.traverse_postorder() if not isinstance(leaf, CompoundModel))
    return [(key, node(inp))]


def _is_linear(model):
    """`True` if ``model`` is linear in its input, so it can act on each component separately."""
    if isinstance(model, CompoundModel):
        return model.op in ("|", "+") and _is_linear(model.left) and _is_linear(model.right)
    return isinstance(model, (MatrixModel, Albedo))


def _key_label(key):
    if key == (_REFLECTED,):
        return DISPLAY_NAMES["Albedo"]
    named = [label for label, is_modifier in key if not is_modifier]
    return _JOIN.join(named or [label for label, _ in key]) or "Other"


def model_components(model, x, names=None):
    """
    Split ``model`` into its additive components, evaluated at ``x``.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The (fitted) model.
    x : `~astropy.units.Quantity`
        The input the model is evaluated on (e.g. photon energy edges).
    names : `dict`, optional
        Display names by model name, overriding `DISPLAY_NAMES`.

    Returns
    -------
    `dict` [`str`, `~astropy.units.Quantity`]
        Component values by name, in the order they appear in the model with
        any albedo reflection last. They add up to ``model(x)``.
    """
    labels = _leaf_labels(model, names)
    components = OrderedDict()
    for key, value in _terms(model, x, labels):
        label = _key_label(key)
        components[label] = components[label] + value if label in components else value
    if DISPLAY_NAMES["Albedo"] in components:
        # The reflected part follows the components it is reflected from.
        components.move_to_end(DISPLAY_NAMES["Albedo"])

    total = model(x)
    summed = sum(components.values())
    if not np.allclose(u.Quantity(summed).value, u.Quantity(total).value, rtol=1e-6, atol=0, equal_nan=True):
        warnings.warn(
            "The model components do not add up to the total; the model may contain a non-linear step.",
            skip_file_prefixes=_WARN_SKIP,
        )
    return components


def _palette(colours):
    """A sequence of colours: as given, or shades of a single colour."""
    if is_color_like(colours):
        return (colours, _darken(colours, 0.4), _lighten(colours, 0.45))
    return tuple(colours)


def _component_styles(model, labels, colours, names):
    """
    ``{label: colour}``: each kind of model in its colour, further components
    of a kind in the next colours of its sequence. Components are always drawn
    solid; line styles tell spectra apart.
    """
    leaf_labels = _leaf_labels(model, names)
    kind_of = {label: type(leaf).__name__ for leaf in _leaves(model) for label in [leaf_labels[id(leaf)][0]]}
    kind_of[DISPLAY_NAMES["Albedo"]] = "Albedo"

    spare = iter(_SPARE_COLOURS)
    palettes, uses, styles = {}, {}, {}
    for label in labels:
        if is_color_like(colours.get(label)):
            # A colour for this component (a sequence under its name is for its kind).
            styles[label] = colours[label]
            continue
        kind = kind_of.get(label.split(_JOIN)[0], label)
        if kind not in palettes:
            given = colours.get(kind)
            if given is None:
                given = COMPONENT_COLOURS.get(kind)
            palettes[kind] = _palette(given if given is not None else next(spare, "C7"))
        palette = palettes[kind]
        # Kinds sharing a sequence (thick and thin target) take its colours in turn.
        repeat = uses.get(palette, 0)
        uses[palette] = repeat + 1
        styles[label] = palette[repeat % len(palette)]
    return styles


# ---------------------------------------------------------------------- #
# Parameters
# ---------------------------------------------------------------------- #
def _free_parameters(model):
    return [n for n in model.param_names if not model.fixed[n] and not model.tied[n]]


def _uncertainties(model):
    stds = getattr(model, "stds", None)
    if stds is None:
        return {}
    try:
        return dict(zip(stds.param_names, stds.stds))
    except (AttributeError, TypeError):
        return {}


def _format_value(value, error):
    """``value ± error`` to the error's precision, in scientific notation when large or small."""
    has_error = error is not None and np.isfinite(error) and error > 0
    exponent = 0
    if value != 0 and not 1e-3 <= abs(value) < 1e4:
        exponent = math.floor(math.log10(abs(value)))
        value, error = value / 10**exponent, (error / 10**exponent if has_error else error)
    if has_error:
        decimals = min(6, max(0, 1 - math.floor(math.log10(error))))
        text = rf"{value:.{decimals}f} $\pm$ {error:.{decimals}f}"
        if exponent:
            text = f"({text})"
    else:
        text = f"{value:.3g}"
    return rf"{text} $\times 10^{{{exponent}}}$" if exponent else text


def _parameter_records(model, component_labels, overrides, names):
    """
    One record per free or tied parameter: the component (or scaling) it
    belongs to, its display symbol, unit, value and uncertainty.
    """
    labels = _leaf_labels(model, names)
    leaves = _leaves(model)
    errors = _uncertainties(model)
    param_map = getattr(model, "_param_map", None)
    records = []
    for name in model.param_names:
        if model.fixed[name]:
            continue
        index, short = param_map[name] if param_map else (0, name)
        leaf = leaves[index]
        cls = type(leaf).__name__
        symbol, unit = (
            overrides.get(name) or PARAMETER_LABELS.get((cls, short)) or PARAMETER_LABELS.get(short, (None, None))
        )
        param = getattr(model, name)
        if unit is None:
            unit = "" if param.unit is None else param.unit.to_string("latex_inline")

        leaf_label, is_modifier = labels[id(leaf)]
        if isinstance(leaf, Albedo):
            group = DISPLAY_NAMES["Albedo"]
        elif is_modifier:
            # Scalings (distance, constants) are listed under their own name.
            group = leaf_label
        else:
            group = next((c for c in component_labels if leaf_label in c.split(_JOIN)), leaf_label)
        records.append(
            {
                "group": group,
                "short": short,
                "symbol": f"${symbol}$" if symbol else short.replace("_", " "),
                "unit": unit,
                "value": float(param.value),
                "error": errors.get(name),
                "tied": bool(model.tied[name]),
            }
        )
    return records


def _parameter_text(record):
    text = f"{record['symbol']} = {_format_value(record['value'], record['error'])} {record['unit']}"
    return text.rstrip()


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
# How each spectrum is told apart when several share one plot: its marker, and
# the line style of its model and components. Dotted and dash-dot lines are kept
# for this; within one spectrum every line is solid.
_DATASET_LOOKS = [
    {"marker": "o", "open": False, "linestyle": "-", "shade": "0.15"},
    {"marker": "s", "open": True, "linestyle": ":", "shade": "0.35"},
    {"marker": "^", "open": True, "linestyle": "-.", "shade": "0.45"},
    {"marker": "D", "open": True, "linestyle": "--", "shade": "0.55"},
    {"marker": "v", "open": True, "linestyle": (0, (4, 1.5, 1, 1.5, 1, 1.5)), "shade": "0.6"},
]
# The reflected albedo is drawn thinner, fainter and underneath the other components.
_ALBEDO_LOOK = {"width": 0.6, "alpha": 0.6, "zorder": 1}
_LAYOUTS = ("separate", "combined")
_HIGHLIGHTS = ("auto", "band", "halo", "bar", None)
_LEGENDS = ("outside", "inside", None)
#: Colour marking the fitted energy range.
HIGHLIGHT_COLOUR = "#ffe680"
# Data outside the fitted range are drawn faded.
_UNFITTED_ALPHA = 0.3
# Opacity of the data dots' fill, and the size of the error bar caps (points).
_DOT_ALPHA = 0.6
_CAP_SIZE = 1.5
# Size and colour of the observation time above a plot: a little larger and darker than the grey details.
_TIME_SIZE = "medium"
_TIME_COLOUR = "0.2"
# Gap between the plots and the summary box beside them, in inches.
_KEY_GAP = 0.12
# Where a key inside the plot may go, in order of preference; the upper ones can
# be cleared by raising the top of the count axis.
_INSIDE_LOCATIONS = (
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "upper center",
    "lower center",
    "center right",
    "center left",
)
_UPPER_LOCATIONS = ("upper right", "upper left", "upper center")
# The tallest an inside key may be, as a fraction of the spectrum panel; the figure is made taller for one
# that is taller still, so that there is room to raise the spectra out of its way.
_MAX_KEY_HEIGHT = 0.55
# Clearance kept around an inside key, as a fraction of the axes.
_KEY_CLEARANCE = 0.012


def plot_fit(
    fit_manager,
    *,
    layout="separate",
    y="flux",
    fitted=True,
    show_components=True,
    show_parameters=True,
    colours=None,
    names=None,
    labels=None,
    parameter_labels=None,
    xscale="log",
    title=None,
    figsize=None,
    show_all=True,
    highlight="auto",
    grid=None,
    legend="outside",
    show_time=True,
):
    """
    Plot fitted spectra, model components and residuals.

    Works with any model combination: the fitted model is split into its
    additive components (see `model_components`). The total model and each
    component are drawn as solid step lines over the count bin edges, the
    total in black and each component in its colour, with the residuals
    underneath. Several components of one kind (two thermal components, say)
    take shades of that kind's colour. When the model has a single component,
    that component is drawn as the model in its own colour.

    Parameters
    ----------
    fit_manager : `~sunkit_spex.fitting.fitter.FitManager`
        A fit manager, normally after `~sunkit_spex.fitting.fitter.FitManager.fit`.
    layout : {"separate", "combined"}, optional
        For joint fits: one panel per spectrum (default), sharing both axes
        with labels on the left and each spectrum named above its panel; or all
        spectra on the same axes, each with its own marker and line style
        (solid, dotted, dash-dot, dashed, ...) for its model and components.
    y : {"flux", "rate", "counts"}, optional
        Show counts per second per keV (default), counts per second, or counts.
    fitted : `bool`, optional
        Plot the fitted models (default) or the initial ones.
    show_components, show_parameters : `bool`, optional
        Draw the individual components; list the fitted parameter values (with
        their uncertainties) in the key, inside or outside. Every model with
        free parameters is listed, components and scalings (a calibration
        constant, say) alike; in a joint fit, a model or value belonging to
        only some spectra is marked with them.
    colours : `dict`, optional
        Colours by component name or model class name, overriding
        `COMPONENT_COLOURS`. A class name can be given one colour, whose
        shades are used for further components of the class, or a sequence.
    names : `dict`, optional
        Display names by model name, overriding `DISPLAY_NAMES`, e.g.
        ``{"ThermalEmission": "Hot plasma"}``.
    labels : `list` [`str`], optional
        Names of the spectra in a joint fit. Defaults to ``meta["instrument"]``
        or "Spectrum 1", "Spectrum 2", ...
    parameter_labels : `dict`, optional
        ``{parameter name: (symbol, unit string)}`` overriding the defaults.
    xscale : {"log", "linear"}, optional
        Energy axis scale.
    title : `str`, optional
        Title, left-aligned above the plot (the whole figure for separate
        joint plots).
    figsize : `tuple`, optional
        Figure size; by default 8 x 6 inches for one panel, plus room for the
        summary box.
    show_all : `bool`, optional
        Plot every bin of each spectrum (default), with the data outside the
        fit range faded and left out of the chi-squared; or only the fitted bins.
    highlight : {"auto", "band", "halo", "bar", None}, optional
        How the fitted energy range is marked: a light band behind it in both
        panels, a halo around the fitted data and model, a bar along the
        bottom of the residual panel, or not at all. ``"auto"`` (default)
        uses the band, or the halo when spectra with different fit ranges
        share a plot (``layout="combined"``), where bands would overlap.
    grid : `tuple` (rows, columns), optional
        Arrangement of the panels in a separate joint layout. Either number
        can be `None` to work it out from the other. By default one row is
        used for up to three spectra, and a near-square grid for more.
    legend : {"outside", "inside", None}, optional
        ``"outside"`` (default): a box beside the plots, its top level with
        theirs, with the reduced chi-squared, each model with its fitted
        parameters and, for joint fits, each spectrum with its reduced
        chi-squared. ``"inside"``: a key inside the plot with each model and
        its fitted parameters, placed where it covers no data or model lines,
        raising the top of the count axis to make room if there is no such
        place; the reduced chi-squared goes above the plot. Grids
        of panels are too small for this and always get the box. `None`: no
        key. The key also gives the fitted energy range of each spectrum.
    show_time : `bool`, optional
        Show each spectrum's observation time range (UTC), from
        ``meta["time_range"]`` (the key can be changed with the FitManager's
        ``meta_keys``): on the line above a single plot, after any title;
        over the fit range and chi-squared of each panel of a joint fit; or,
        for spectra sharing a plot, under each one's name in the key. Spectra
        need not cover the same time.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
    axes : `numpy.ndarray` [`~matplotlib.axes.Axes`]
        Shape ``(2, n_panels)``: spectrum axes on the first row, residual
        axes on the second, one column per panel (per spectrum, in order, in
        a separate joint layout).
    """
    if y not in _Y_UNITS:
        raise ValueError(f"Unknown y {y!r}; choose from {', '.join(_Y_UNITS)}.")
    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; choose from {', '.join(_LAYOUTS)}.")
    if highlight not in _HIGHLIGHTS:
        raise ValueError(f"Unknown highlight {highlight!r}; choose from {', '.join(map(str, _HIGHLIGHTS))}.")
    if legend not in _LEGENDS:
        raise ValueError(f"Unknown legend {legend!r}; choose from {', '.join(map(str, _LEGENDS))}.")
    options = {
        "colours": colours or {},
        "names": names or {},
        "parameter_labels": parameter_labels or {},
        "show_components": show_components,
        "show_parameters": show_parameters,
        "highlight": highlight,
    }

    spectra = fit_manager.fit_spectra
    models = fit_manager.fitted_models if fitted else _with_ties_applied(fit_manager.prepared_models)
    n = len(spectra)
    labels = list(labels) if labels is not None else [_spectrum_name(s, i) for i, s in enumerate(spectra)]
    datasets = []
    for i, (model, spectrum, name) in enumerate(zip(models, spectra, labels)):
        fit_edges = u.Quantity(spectrum.spectral_axis.bin_edges)
        full = _full_spectrum(fit_manager, i, model) if show_all else None
        if full is not None:
            model, spectrum = full
        d = _prepare_dataset(fit_manager, model, spectrum, y, options, fit_edges)
        time = _time_text(fit_manager, fit_manager.spectra[i]) if show_time else None
        datasets.append(d | {"name": name, "index": i, "time": time})
    _mark_differences(datasets)
    _share_degrees_of_freedom(datasets, models)
    combined = layout == "combined" and n > 1
    groups = [datasets] if layout == "combined" or n == 1 else [[d] for d in datasets]
    n_panels = len(groups)
    rows, columns = _grid_shape(n_panels, grid)
    many_panels = n_panels > 1 and (rows > 1 or columns > 3)
    if legend == "inside" and many_panels:
        legend = "outside"  # small panels have no room for a key

    rc = RC_PARAMS | (_SMALL_PANEL_RC if many_panels else {})
    beside = legend == "outside"
    with plt.rc_context(rc):
        if many_panels:
            panel_width, height = _GRID_CELL[0], rows * _GRID_CELL[1] + 0.8
        else:
            panel_width, height = _FIGSIZE[0] * (1 if n_panels == 1 else 0.7), _FIGSIZE[1]
        fig = plt.figure(figsize=figsize or (columns * panel_width, height), layout="constrained")
        # Each panel is a spectrum over its residuals; spacer rows keep the panels apart. The
        # summary box gets a column of its own, so its top lines up with the top of the plots.
        ratios = [r for row in range(rows) for r in ([3, 1] + ([_GRID_GAP] if row < rows - 1 else []))]
        outer = fig.add_gridspec(len(ratios), columns + beside, height_ratios=ratios)
        axes = np.empty((2, n_panels), dtype=object)

        keys = []
        for panel, group in enumerate(groups):
            row, column = divmod(panel, columns)
            # Separate panels share both axes, so the spectra compare directly.
            first, first_res = axes[0, 0], axes[1, 0]
            ax = fig.add_subplot(outer[3 * row, column], sharex=first, sharey=first)
            ax_res = fig.add_subplot(outer[3 * row + 1, column], sharex=ax, sharey=first_res)
            axes[:, panel] = ax, ax_res
            group_options = options | {"highlight": _resolve_highlight(highlight, group)}
            keys += _draw_group(ax, ax_res, group, group_options, xscale, combined=len(group) > 1)

        overall = _chi2_summary(datasets, models)
        heading = title
        if n_panels > 1:
            _share_panels(fig, axes, datasets, rows, columns)
            # Each panel is named above it, with its time, its fit range (when they differ) and its
            # chi-squared: nothing is written over the data.
            for ax, d in zip(axes[0], datasets):
                ax.set_title(d["name"], loc="left", fontweight="bold")
                details = " \N{MIDDLE DOT} ".join(filter(None, [d["own_range"], _chi2_text(d)]))
                ax.set_title(details, loc="right", fontsize="small", color="0.35")
                if d["time"]:
                    _time_above_right_title(ax, d["time"])
            if legend is None:
                heading = f"{heading}   {overall}" if heading else overall
            if heading:
                fig.suptitle(heading, x=0.01, ha="left", fontsize=rc["axes.titlesize"])
        else:
            # One line above the plot: the title and the time on the left, the reduced chi-squared
            # on the right. Spectra sharing a plot have their times in the key.
            time = datasets[0]["time"] if not combined else None
            left = "   ".join(filter(None, [heading, time]))
            if left:
                size = rc["axes.titlesize"] if heading else _TIME_SIZE
                axes[0, 0].set_title(left, loc="left", fontsize=size, color="0.1" if heading else _TIME_COLOUR)
            right = []
            if combined and legend is None:
                right = [f"{d['name']}: {d['time']}" for d in datasets if d["time"]]
            if legend != "outside":
                right.append(overall)
            axes[0, 0].set_title("\n".join(right), loc="right", fontsize="small", color="0.35")

        if beside:
            # The key hangs from the top of the first row of plots. Its column is sized to it, so
            # it is left out of the layout (an axes spanning the rows would space them apart).
            key_axes = fig.add_subplot(outer[0, -1])
            key_axes.set_axis_off()
            entries = _key_entries(keys, combined=combined, full=True)
            key = _draw_key(key_axes, entries, title=overall, combined=combined, summary=True)
            key.set_in_layout(False)
            _size_key_column(fig, outer, key, key_axes, axes[0, 0], columns, panel_width, keep_size=figsize is not None)
        elif legend == "inside":
            entries = _key_entries(keys, combined=combined, full=False)
            _place_key(
                fig,
                axes[0],
                groups,
                keys,
                entries,
                title=overall if n_panels > 1 else None,
                combined=combined,
                keep_size=figsize is not None,
            )

    return fig, axes


def _chi2_summary(datasets, models):
    """The (joint) reduced chi-squared: chi-squared over fitted bins minus free parameters."""
    chi2 = sum(d["chi2"] for d in datasets)
    dof = sum(d["n_bins"] for d in datasets) - sum(len(_free_parameters(m)) for m in models)
    text = rf"$\chi^2_\mathrm{{red}}$ = {chi2 / dof:.2f}" if dof > 0 else rf"$\chi^2$ = {chi2:.1f}"
    return f"Joint {text}" if len(datasets) > 1 else text


def _share_degrees_of_freedom(datasets, models):
    """
    Each spectrum's degrees of freedom, ``dof``: its fitted bins less its share
    of the free parameters, in proportion to its bins (shared parameters can't
    be given to one spectrum). They add up to the joint degrees of freedom.
    """
    n_free = sum(len(_free_parameters(m)) for m in models)
    n_bins = sum(d["n_bins"] for d in datasets)
    for d in datasets:
        d["dof"] = d["n_bins"] * (1 - n_free / n_bins) if n_bins else 0


def _time_above_right_title(ax, time):
    """The time over the panel's right title (its fit range and chi-squared), a little larger."""
    small = FontProperties(size="small").get_size_in_points()
    # From the axes top: the title's pad, then the height of the right title, sub- and superscripts and all.
    rise = plt.rcParams["axes.titlepad"] + 1.55 * small
    ax.text(
        1,
        1,
        time,
        transform=offset_copy(ax.transAxes, fig=ax.figure, y=rise, units="points"),
        ha="right",
        va="baseline",
        fontsize=_TIME_SIZE,
        color=_TIME_COLOUR,
    )


def _format_energy(value):
    return f"{value:.3g}"


def _time_text(fit_manager, spectrum):
    """The spectrum's observation time range as text, or `None` if its meta has none."""
    key = fit_manager._meta_keys.get("time_range", "time_range")
    value = spectrum.meta.get(key)
    if value is None:
        return None
    try:
        if hasattr(value, "start") and hasattr(value, "end"):  # e.g. a sunpy TimeRange
            start, end = Time(value.start), Time(value.end)
        else:
            times = Time(value).ravel()
            start, end = times.min(), times.max()
    except (TypeError, ValueError):
        return str(value)
    return _format_time_range(start, end)


def _format_time_range(start, end):
    """``2024-03-10 12:05:50\N{EN DASH}12:05:55`` (UTC), with the date once if both ends share it."""

    def parts(time):
        date, clock = time.utc.isot.split("T")
        return date, clock.rstrip("0").rstrip(".")

    (start_date, start_clock), (end_date, end_clock) = parts(start), parts(end)
    if start_date == end_date:
        return f"{start_date} {start_clock}\N{EN DASH}{end_clock}"
    return f"{start_date} {start_clock} \N{EN DASH} {end_date} {end_clock}"


def _mark_differences(datasets):
    """``own_range``: each spectrum's fit range when they differ (else `None`, as it is shown once for all)."""
    ranges = {d["fit_text"] for d in datasets}
    for d in datasets:
        d["own_range"] = d["fit_text"] if len(ranges) > 1 else None


def _chi2_text(d):
    """One spectrum's reduced chi-squared (see `_share_degrees_of_freedom`)."""
    if d["dof"] > 0:
        return rf"$\chi^2_\mathrm{{red}}$ = {d['chi2'] / d['dof']:.2f}"
    return rf"$\chi^2$ = {d['chi2']:.1f}"


def _with_ties_applied(models):
    """
    Copies of the (initial) models with tied parameters set from what they are
    tied to; the fitters only do this while fitting.
    """
    models = [copy.deepcopy(model) for model in models]
    for model in models:
        for name in model.param_names:
            tie = model.tied[name]
            if tie:
                # Joint ties take the list of models; a single model's ties take the model.
                value = tie(models if len(models) > 1 else model)
                if isinstance(value, Parameter):
                    value = value.value if value.unit is None else value.quantity
                setattr(model, name, value)
    return models


def _resolve_highlight(highlight, group):
    if highlight != "auto":
        return highlight
    ranges = {tuple(np.round(d["fit_range"], 9)) for d in group}
    return "halo" if len(ranges) > 1 else "band"


def _grid_shape(n_panels, grid):
    """``(rows, columns)`` for ``n_panels``: as given, filled in, or one row up to three then near-square."""
    rows, columns = (None, None) if grid is None else grid
    if rows is None and columns is None:
        if n_panels <= 3:
            return 1, n_panels
        columns = math.ceil(math.sqrt(n_panels))
    if rows is None:
        rows = math.ceil(n_panels / columns)
    elif columns is None:
        columns = math.ceil(n_panels / rows)
    if rows < 1 or columns < 1 or rows * columns < n_panels:
        raise ValueError(f"A {rows} x {columns} grid can't hold {n_panels} panels.")
    return rows, columns


def _share_panels(fig, axes, datasets, rows, columns):
    """
    Common limits for the shared axes; y labels on the left column, energy tick
    labels on the bottom of each column and one energy label.
    """
    n_panels = axes.shape[1]
    axes[0, 0].set_ylim(*_count_limits(datasets))
    axes[1, 0].set_ylim(*_residual_limits(datasets))
    axes[1, 0].set_xlim(*_energy_limits(datasets))

    for panel, ax_res in enumerate(axes[1]):
        column = panel % columns
        ax_res.set_xlabel("")
        if column > 0:
            for axis in axes[:, panel]:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False)
        if panel + columns < n_panels:
            # Another panel below: its energy axis is labelled instead.
            ax_res.tick_params(labelbottom=False)

    fig.supxlabel(f"Energy ({datasets[0]['energy_unit'].to_string('latex_inline')})", fontsize="medium")
    if rows > 1:
        # Spectrum and residuals touch; the spacer rows separate the panels.
        fig.get_layout_engine().set(wspace=0.01, w_pad=0.02, hspace=0, h_pad=0)
    else:
        fig.get_layout_engine().set(wspace=0.01, w_pad=0.02)


def _count_limits(datasets):
    positive = np.concatenate([d["data"][d["data"] > 0] for d in datasets])
    floor = 0.5 * positive.min() if positive.size else 1e-3
    top = 2 * max(positive.max() if positive.size else 1, max(d["total"].max() for d in datasets))
    return floor, top


def _residual_limits(datasets):
    """Symmetric limits set by the fitted bins; unfitted bins far off the model fall outside."""
    finite = np.abs(
        np.concatenate([d["residuals"][np.isfinite(d["residuals"]) & d.get("fitted", True)] for d in datasets])
    )
    limit = max(3.5, 1.15 * finite.max()) if finite.size else 3.5
    return -limit, limit


def _energy_limits(datasets):
    edges = np.concatenate([d["edges"] for d in datasets])
    return edges.min(), edges.max()


def _full_spectrum(fit_manager, index, model):
    """
    ``(model, spectrum)`` covering every bin of the original spectrum, with the
    fitted parameter values; `None` if the response can't be rebuilt for it.
    """
    user_leaves = _leaves(fit_manager.models[index])
    model = copy.deepcopy(model)
    for user_leaf, leaf in zip(user_leaves, _leaves(model)):
        if isinstance(leaf, MatrixModel):
            if not user_leaf.is_placeholder:
                # A response given by the user only covers the bins it was made for.
                return None
            leaf.matrix = leaf.photon_edges = leaf.area = leaf.exposure_time = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # already warned about when preparing the fit
        return fit_manager._prepare(model, fit_manager.spectra[index], None)


def _spectrum_name(spectrum, index):
    return str(spectrum.meta.get("instrument") or f"Spectrum {index + 1}")


def _prepare_dataset(fit_manager, model, spectrum, y, options, fit_edges):
    """Everything needed to draw one spectrum and its model, in the requested ``y`` units."""
    edges = u.Quantity(spectrum.spectral_axis.bin_edges)
    # Bins that were fitted: those lying within the fitted bin edges.
    low, high = fit_edges[0].to_value(edges.unit), fit_edges[-1].to_value(edges.unit)
    tolerance = 1e-9 * abs(high)
    fitted = (edges.value[:-1] >= low - tolerance) & (edges.value[1:] <= high + tolerance)
    counts = u.Quantity(spectrum.data, spectrum.unit).to(u.ct)
    sigma = None if spectrum.uncertainty is None else u.Quantity(spectrum.uncertainty.array, u.ct)

    components = model_components(model, fit_manager._model_x(model, spectrum), options["names"])
    total = sum(components.values()).to(u.ct)

    per_time, per_energy, ylabel = _Y_UNITS[y]
    factor = 1 * u.dimensionless_unscaled
    if per_time:
        factor = factor / fit_manager._meta(spectrum, "exposure_time", required_for=f"plotting y={y!r}")
    if per_energy:
        factor = factor / np.diff(edges)
    yunit = (counts * factor).unit

    if sigma is None:
        residuals = (counts - total).value
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = np.where(sigma.value > 0, ((counts - total) / sigma).to_value(""), np.nan)
    good = np.isfinite(residuals) & (sigma is not None) & fitted

    return {
        "model": model,
        "edges": edges.value,
        "energy_unit": edges.unit,
        "centres": u.Quantity(spectrum.spectral_axis).value,
        "data": (counts * factor).to_value(yunit),
        "error": None if sigma is None else (sigma * factor).to_value(yunit),
        "total": (total * factor).to_value(yunit),
        "components": OrderedDict((k, (v.to(u.ct) * factor).to_value(yunit)) for k, v in components.items()),
        "styles": _component_styles(model, list(components), options["colours"], options["names"]),
        "records": (
            _parameter_records(model, list(components), options["parameter_labels"], options["names"])
            if options["show_parameters"]
            else {}
        ),
        "ylabel": ylabel,
        "residuals": residuals,
        "has_sigma": sigma is not None,
        "chi2": float(np.sum(residuals[good] ** 2)) if sigma is not None else 0.0,
        "n_bins": int(good.sum()) if sigma is not None else int(fitted.sum()),
        "fitted": fitted,
        "fit_range": (low, high),
        "fit_text": f"{_format_energy(low)}\N{EN DASH}{_format_energy(high)} {edges.unit.to_string()}",
    }


def _lighten(colour, amount):
    rgb = np.array(to_rgb(colour))
    return tuple(rgb + (1 - rgb) * amount)


def _darken(colour, amount):
    return tuple(np.array(to_rgb(colour)) * (1 - amount))


def _draw_group(ax, ax_res, group, options, xscale, combined):
    """Draw one or more spectra (``combined``) on a spectrum and a residual axis; returns what the key needs."""
    keys = []
    for d in group:
        look = _DATASET_LOOKS[d["index"] % len(_DATASET_LOOKS)] if combined else _DATASET_LOOKS[0]
        d["look"] = look
        data_handle = _draw_data(ax, d, look, combined)
        linestyle = look["linestyle"] if combined else "-"

        single = len(d["components"]) == 1
        if single:
            # One component is the whole model: draw it binned, in its own colour.
            (label,) = d["components"]
            model_handle = ax.stairs(
                d["total"],
                d["edges"],
                baseline=None,
                color=d["styles"][label],
                lw=_TOTAL_WIDTH,
                ls=linestyle,
                zorder=5,
            )
            # Alone, it heads the key with its line; with several spectra each has its own line above.
            component_handles = {} if combined else {label: model_handle}
        else:
            model_handle = ax.stairs(
                d["total"], d["edges"], baseline=None, color=TOTAL_COLOUR, lw=_TOTAL_WIDTH, ls=linestyle, zorder=5
            )
            component_handles = _draw_components(ax, d, look, combined) if options["show_components"] else {}
        _draw_residuals(ax_res, d, look, combined)
        highlight = _draw_highlight(ax, ax_res, d, options["highlight"], combined, list(component_handles))
        keys.append((d, data_handle, model_handle, component_handles, single, highlight))

    ax.set_yscale("log")
    ax.set_ylim(*_count_limits(group))
    ax.set_ylabel(group[0]["ylabel"])
    ax.tick_params(labelbottom=False)

    ax_res.set_xlim(*_energy_limits(group))
    _set_energy_axis(ax_res, xscale)
    _finish_residual_axis(ax_res, group)
    return keys


def _points(ax, x, y, error, look, combined, mask, edges=None, **kwargs):
    """
    Data-style points with error bars, faded where ``mask`` is False; returns the
    fitted ones. With ``edges``, horizontal bars span each bin.
    """
    colour = "0.2" if not combined else look["shade"]
    # Slightly see-through dots with a solid outline.
    face = (1, 1, 1, _DOT_ALPHA) if look["open"] else to_rgba(colour, _DOT_ALPHA)
    xerr = None if edges is None else np.vstack([x - edges[:-1], edges[1:] - x])
    yerr = None if error is None else np.broadcast_to(error, y.shape)
    handle = None
    for selected, faded in ((mask, False), (~mask, True)):
        if not selected.any():
            continue
        artist = ax.errorbar(
            x[selected],
            y[selected],
            xerr=None if xerr is None else xerr[:, selected],
            yerr=None if yerr is None else yerr[selected],
            fmt=look["marker"],
            ms=3.5,
            mfc=face,
            mec=colour,
            mew=0.9,
            color=colour,
            ecolor="0.55",
            elinewidth=0.9,
            capsize=_CAP_SIZE,
            capthick=0.9,
            # Only fade the unfitted points: an alpha would override the dots' own.
            alpha=_UNFITTED_ALPHA if faded else None,
            **kwargs,
        )
        if not faded:
            handle = artist
    return handle


def _draw_data(ax, d, look, combined):
    return _points(ax, d["centres"], d["data"], d["error"], look, combined, d["fitted"], edges=d["edges"], zorder=2)


def _draw_highlight(ax, ax_res, d, style, combined, drawn_components):
    """Mark the fitted energy range; returns a key handle, or `None`."""
    if style is None or not d["fitted"].any():
        return None
    low, high = d["fit_range"]

    if style == "band":
        for axis in (ax, ax_res):
            axis.axvspan(low, high, color=HIGHLIGHT_COLOUR, alpha=0.35, lw=0, zorder=0)
        return Patch(facecolor=HIGHLIGHT_COLOUR, alpha=0.35, lw=0)

    if style == "bar":
        # A strip along the bottom of the residual panel; stacked when spectra share a plot.
        height = 0.06
        bottom = d["index"] * height * 1.15 if combined else 0
        # Energy along x, axes fraction up y.
        transform = blended_transform_factory(ax_res.transData, ax_res.transAxes)
        strip = Rectangle(
            (low, bottom),
            high - low,
            height,
            transform=transform,
            facecolor=HIGHLIGHT_COLOUR,
            lw=0,
            zorder=0.6,
        )
        ax_res.add_patch(strip)
        if combined:
            # Say which spectrum each bar belongs to.
            ax_res.text(
                high,
                bottom + height / 2,
                f" {d['name']}",
                transform=transform,
                ha="left",
                va="center",
                fontsize="x-small",
                color="0.3",
            )
        return Patch(facecolor=HIGHLIGHT_COLOUR, lw=0)

    # "halo": a soft glow behind the fitted data and the model over the fitted range.
    fitted = d["fitted"]
    first, last = np.flatnonzero(fitted)[[0, -1]]
    edges = d["edges"][first : last + 2]
    glow = {"color": HIGHLIGHT_COLOUR, "alpha": 0.8, "zorder": 0.8}
    ax.stairs(d["total"][first : last + 1], edges, baseline=None, lw=7, **glow)
    for label in drawn_components:
        ax.stairs(d["components"][label][first : last + 1], edges, baseline=None, lw=6, **glow)
    for axis, values in ((ax, d["data"]), (ax_res, d["residuals"])):
        axis.plot(d["centres"][fitted], values[fitted], ls="none", marker="o", ms=9, mec="none", **glow)
    return Line2D([], [], lw=7, **glow)


def _draw_components(ax, d, look, combined):
    """Each component as a step line over the count bin edges, in its colour and its spectrum's line style."""
    handles = {}
    linestyle = look["linestyle"] if combined else "-"
    for label, values in d["components"].items():
        colour = d["styles"][label]
        width, alpha, zorder = _COMPONENT_WIDTH, 1.0, 3
        if label == DISPLAY_NAMES["Albedo"]:
            width, alpha, zorder = width * _ALBEDO_LOOK["width"], _ALBEDO_LOOK["alpha"], _ALBEDO_LOOK["zorder"]
        handles[label] = ax.stairs(
            values, d["edges"], baseline=None, color=colour, ls=linestyle, lw=width, alpha=alpha, zorder=zorder
        )
    return handles


def _draw_residuals(ax_res, d, look, combined):
    error = 1.0 if d["has_sigma"] else None
    _points(ax_res, d["centres"], d["residuals"], error, look, combined, d["fitted"], edges=d["edges"])


def _finish_residual_axis(ax_res, group):
    ax_res.axhline(0, color="0.3", lw=0.9, zorder=0)
    ax_res.set_ylim(*_residual_limits(group))
    ax_res.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, symmetric=True))
    ax_res.set_ylabel(r"(D $-$ M) / $\sigma$" if group[0]["has_sigma"] else "D $-$ M (ct)")
    ax_res.set_xlabel(f"Energy ({group[0]['energy_unit'].to_string('latex_inline')})")


def _summary(keys):
    """
    What the key says about the model, across the spectra drawn: one group per
    model, each a `dict` with its ``label``, ``colour``, ``handle``, the
    ``spectra`` it is in, and its parameter ``lines``.

    Every model is listed: the components drawn, and any other model with free
    parameters (a calibration constant, a distance). In a joint fit a model in
    only some spectra is marked with them. A parameter is shared when it is
    free in one spectrum and tied, with the same value, in every other
    spectrum the model is in: it is listed once. Otherwise each value is
    marked with the spectra it belongs to.
    """
    names = [key[0]["name"] for key in keys]
    groups = OrderedDict()
    for i, (d, _, _, handles, _, _) in enumerate(keys):
        for label in d["components"]:
            group = groups.setdefault(label, _new_group(label, d, is_component=True))
            group["spectra"].append(i)
            group["handle"] = group["handle"] or handles.get(label)
        for record in d["records"]:
            label = record["group"]
            group = groups.setdefault(label, _new_group(label, d, is_component=label in d["components"]))
            if i not in group["spectra"]:
                group["spectra"].append(i)
            group["params"].setdefault(record["short"], {})[i] = record

    for group in groups.values():
        for records in group["params"].values():
            for members, record in _shared_values(records):
                text = _parameter_text(record)
                if set(members) != set(group["spectra"]):
                    text += _spectra_tag(members, names)
                group["lines"].append(text)
        if len(group["spectra"]) < len(keys):
            group["label"] += _spectra_tag(group["spectra"], names)
    return list(groups.values())


def _spectra_tag(members, names):
    """``   [A, B]`` naming the spectra ``members``; shortened when there are many."""
    members = sorted(members)
    if len(members) <= 3:
        inner = ", ".join(names[m] for m in members)
    elif len(names) - len(members) <= 2:
        inner = "all but " + ", ".join(names[m] for m in range(len(names)) if m not in members)
    else:
        inner = f"{len(members)} spectra"
    return f"   [{inner}]"


def _new_group(label, d, is_component):
    return {
        "label": label,
        "colour": d["styles"].get(label, "0.15"),
        "handle": None,
        "spectra": [],
        "is_component": is_component,
        "params": OrderedDict(),
        "lines": [],
    }


def _shared_values(records):
    """
    ``[(spectra, record)]``: each free value of a parameter with the spectra
    sharing it (those whose parameter is tied to the same value).
    """
    clusters, assigned = [], set()
    for i, record in records.items():
        if record["tied"]:
            continue
        members = [i] + [
            j
            for j, other in records.items()
            if j != i and j not in assigned and other["tied"] and np.isclose(other["value"], record["value"], rtol=1e-6)
        ]
        assigned.update(members)
        clusters.append((members, record))
    return clusters


def _spectrum_handle(look, colour):
    """A key entry showing a spectrum as it is drawn: its marker on its line style."""
    return Line2D(
        [],
        [],
        color=colour,
        ls=look["linestyle"],
        lw=_TOTAL_WIDTH,
        marker=look["marker"],
        ms=4.5,
        mfc=(1, 1, 1, _DOT_ALPHA) if look["open"] else to_rgba(look["shade"], _DOT_ALPHA),
        mec=look["shade"],
        mew=0.9,
    )


def _swatch(step):
    """A key line for a component: its colour and line style, opaque and wide enough to see."""
    if step is None:
        return None
    colour = to_rgba(step.get_edgecolor(), 1)
    return Line2D([], [], color=colour, ls=step.get_linestyle(), lw=max(step.get_linewidth(), 1.2))


def _key_entries(keys, *, combined, full):
    """
    ``[(handle, text, kind, colour)]`` for the key; ``kind`` is "item", "head"
    (a component), "param" (a parameter value), "section", "note" or "gap".

    Every model is listed with its fitted parameters (when the datasets have
    them, see ``show_parameters``). ``full`` adds section headings and, for
    joint fits, each spectrum's reduced chi-squared.
    """
    blank = Line2D([], [], ls="none")
    entries = []
    groups = _summary(keys)
    several = len(keys) > 1
    highlight = next((key[-1] for key in keys if key[-1] is not None), None)

    def section(name):
        if full and several:
            if entries:
                entries.append((blank, "", "gap", None))
            entries.append((blank, name.upper(), "section", None))

    def spectra(with_handles):
        section("Spectra")
        for d, _, model_handle, _, _, _ in keys:
            handle = _spectrum_handle(d["look"], model_handle.get_edgecolor()) if with_handles else blank
            text = d["name"]
            if d["own_range"]:
                text = f"{text}   {d['own_range']}"
            if full:
                text = f"{text}   {_chi2_text(d)}"
            entries.append((handle, text, "item", "0.15"))
            if d["time"] and with_handles:
                # Spectra sharing a plot: each one's time under its name.
                entries.append((blank, d["time"], "param", "0.35"))

    if combined:
        spectra(with_handles=True)
    else:
        _, data_handle, model_handle, _, single, _ = keys[0]
        entries.append((data_handle, "Data", "item", "0.15"))
        if not single:
            entries.append((model_handle, "Total", "item", "0.15"))
    # The fitted energy range, here when all the spectra share it, else with each spectrum.
    shared_range = "" if keys[0][0]["own_range"] else f"   {keys[0][0]['fit_text']}"
    if highlight is not None:
        entries.append((highlight, f"Fitted range{shared_range}", "item", "0.15"))
    elif shared_range:
        entries.append((blank, f"Fitted range{shared_range}", "item", "0.15"))

    listed = [g for g in groups if g["handle"] is not None or g["lines"]]
    if listed:
        section("Model")
        marked = any("[" in line for g in listed for line in [g["label"], *g["lines"]])
        if several and (marked or any(g["lines"] for g in listed)):
            note = "Shared by all spectra unless marked [spectrum]" if marked else "Shared by all spectra"
            entries.append((blank, note, "note", None))
        for group in listed:
            entries.append((_swatch(group["handle"]) or blank, group["label"], "head", group["colour"]))
            entries.extend((blank, line, "param", group["colour"]) for line in group["lines"])
    if full and several and not combined:
        spectra(with_handles=False)
    return entries


def _draw_key(target, entries, *, title=None, loc="upper right", combined=False, summary=False):
    """
    Draw the key on the axes ``target``: the ``summary`` box, from its top left
    corner, or a compact key at ``loc``.
    """
    outside = summary
    handles = [entry[0] for entry in entries]
    texts = [entry[1] for entry in entries]
    options = {
        "title": title,
        "alignment": "left",
        "frameon": True,
        "fancybox": True,
        "facecolor": "white",
        "edgecolor": "0.82",
        "framealpha": 1 if outside else 0.9,
        # Long enough to show a dotted or dash-dot line.
        "handlelength": 2.8 if combined else 1.8,
        "borderpad": 0.9 if outside else 0.6,
    }
    if outside:
        # Its top on the top of the axes (level with the plots), a little in from the left.
        anchor = offset_copy(target.transAxes, fig=target.figure, x=72 * _KEY_GAP, y=0, units="points")
        legend = target.legend(
            handles, texts, loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=anchor, borderaxespad=0, **options
        )
    else:
        legend = target.legend(handles, texts, loc=loc, fontsize="small", **options)
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_color("0.1")

    small = 0.85 * legend.get_texts()[0].get_fontsize() if entries else None
    for text, (_, _, kind, colour) in zip(legend.get_texts(), entries):
        if colour is not None:
            text.set_color(colour)
        if kind == "head":
            text.set_fontweight("bold")
        elif kind == "param":
            text.set_fontsize(small)
        elif kind == "section":
            text.set_fontsize(0.8 * small)
            text.set_fontweight("bold")
            text.set_color("0.5")
        elif kind == "note":
            text.set_fontsize(0.9 * small)
            text.set_fontstyle("italic")
            text.set_color("0.45")
        elif kind == "gap":
            text.set_fontsize(0.4 * small)
    return legend


def _size_key_column(fig, grid, key, key_axes, panel_axes, columns, panel_width, keep_size):
    """
    Size the summary box's column to the box; without a figure size given, widen
    the figure to hold it (and heighten it if the box is taller than the plots).
    """
    box = key.get_window_extent(fig.canvas.get_renderer())
    need, key_height = box.width / fig.dpi + 2 * _KEY_GAP, box.height / fig.dpi
    width, height = fig.get_size_inches()
    if keep_size:
        panel_width = max((width - need) / columns, 0.5)
    else:
        width = columns * panel_width + need
        fig.set_size_inches(width, max(height, key_height + 1.2))
    grid.set_width_ratios([1] * columns + [need / panel_width])
    # The layout takes margins (axis labels) out of the width first: correct the column to what
    # the box needs once laid out, widening the figure rather than shrinking the plots.
    for _ in range(3):
        fig.draw_without_rendering()
        width = fig.get_size_inches()[0]
        have = key_axes.get_position().width * width
        if have >= need - 0.01:
            break
        if not keep_size:
            fig.set_size_inches(width + need - have, fig.get_size_inches()[1])
            width += need - have
        laid_out_panel = panel_axes.get_position().width * width
        grid.set_width_ratios([1] * columns + [need / max(laid_out_panel, 0.1)])


def _place_key(fig, axes, groups, keys, entries, *, title, combined, keep_size=False):
    """
    Put the key inside a spectrum panel where it covers no data or model line.

    Every corner and edge of every panel is tried. If none is clear, the top
    of the count axis is raised just enough to leave room above the spectra
    in an upper corner. A key too tall for that (over ``_MAX_KEY_HEIGHT`` of the
    panel) first makes the figure taller, unless its size was given.
    """
    fig.draw_without_rendering()  # lays the figure out, so the key's size within each panel is known
    renderer = fig.canvas.get_renderer()
    if not keep_size:
        for _ in range(3):
            legend = _draw_key(axes[0], entries, title=title, combined=combined)
            height = legend.get_window_extent(renderer).height / axes[0].get_window_extent(renderer).height
            legend.remove()
            if height <= _MAX_KEY_HEIGHT:
                break
            width, fig_height = fig.get_size_inches()
            fig.set_size_inches(width, fig_height * height / _MAX_KEY_HEIGHT)
            fig.draw_without_rendering()
    drawn = {id(key[0]): list(key[3]) for key in keys}

    candidates = []  # (covered points, preference, panel, loc, key box, point positions)
    for panel, (ax, group) in enumerate(zip(axes, groups)):
        points = _obstacles(group, drawn)
        fractions = ax.transAxes.inverted().transform(ax.transData.transform(points))
        legend = _draw_key(ax, entries, title=title, combined=combined)
        for order, loc in enumerate(_INSIDE_LOCATIONS):
            legend.set_loc(loc)
            box = legend.get_window_extent(renderer).transformed(ax.transAxes.inverted())
            candidates.append((_covered(fractions, box).sum(), order, panel, loc, box, fractions))
        legend.remove()

    covered, _, panel, loc, _, _ = min(candidates, key=lambda c: c[:3])
    if covered:
        raised = []
        for _, order, candidate_panel, candidate_loc, box, fractions in candidates:
            if candidate_loc not in _UPPER_LOCATIONS:
                continue
            below = box.y0 - _KEY_CLEARANCE
            under = (fractions[:, 0] >= box.x0 - _KEY_CLEARANCE) & (fractions[:, 0] <= box.x1 + _KEY_CLEARANCE)
            if below < 0.3 or not under.any():
                continue  # the key is too tall to sit above the spectra
            # On the log count axis, scaling the span up by `highest / below` brings everything
            # under the key down to its lower edge.
            raised.append((fractions[under, 1].max() / below, order, candidate_panel, candidate_loc))
        if raised:
            scale, _, panel, loc = min(raised)
            bottom, top = axes[panel].get_ylim()
            axes[panel].set_ylim(bottom, bottom * (top / bottom) ** scale)
    return _draw_key(axes[panel], entries, title=title, loc=loc, combined=combined)


def _covered(fractions, box):
    """Which points (in axes fractions) the key box covers, with a little clearance."""
    pad = _KEY_CLEARANCE
    return (
        (fractions[:, 0] >= box.x0 - pad)
        & (fractions[:, 0] <= box.x1 + pad)
        & (fractions[:, 1] >= box.y0 - pad)
        & (fractions[:, 1] <= box.y1 + pad)
    )


def _obstacles(group, drawn):
    """Points along the data, their error bars and the drawn model lines, in data coordinates."""
    points = []
    for d in group:
        edges, x, y = d["edges"], d["centres"], d["data"]
        error = np.zeros_like(y) if d["error"] is None else d["error"]
        for px, py in ((edges[:-1], y), (edges[1:], y), (x, y), (x, y + error), (x, y - error)):
            points.append(np.column_stack([px, py]))
        for values in [d["total"]] + [d["components"][label] for label in drawn.get(id(d), [])]:
            # Along each step...
            points.extend(np.column_stack([edges[:-1] + t * np.diff(edges), values]) for t in np.linspace(0, 1, 5))
            # ...and up or down the risers between them.
            low, high = np.minimum(values[:-1], values[1:]), np.maximum(values[:-1], values[1:])
            rising = low > 0
            points.extend(
                np.column_stack([edges[1:-1][rising], (low ** (1 - t) * high**t)[rising]])
                for t in np.linspace(0, 1, 6)[1:-1]
            )
    points = np.concatenate(points)
    return points[np.isfinite(points).all(axis=1) & (points[:, 1] > 0)]


def _set_energy_axis(ax_res, xscale):
    ax_res.set_xscale(xscale)
    if xscale == "log":
        # Plain energies (5, 10, 20, 50, ...) rather than powers of ten.
        ax_res.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax_res.xaxis.set_major_formatter(ScalarFormatter())
        ax_res.xaxis.set_minor_formatter(NullFormatter())