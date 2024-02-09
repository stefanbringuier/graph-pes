from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from cycler import cycler
from matplotlib.ticker import MaxNLocator

from .core import GraphPESModel
from .data.atomic_graph import AtomicGraph, convert_to_atomic_graphs
from .data.batching import AtomicGraphBatch
from .transform import Identity, Transform
from .util import Property, PropertyKey

_my_style = {
    "figure.figsize": (3.5, 3),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "legend.fancybox": False,
    "savefig.transparent": False,
    "axes.prop_cycle": cycler(
        "color",
        [
            "#4385be",
            "#ce5d97",
            "#da702c",
            "#8b7ec8",
            "#879a39",
            "#d14d41",
            "#d14d41",
            "#d0a215",
            "#3aa99f",
        ],
    ),
}


@contextmanager
def nice_style():
    """
    Context manager to use my home-made plt style within a context
    """
    with plt.rc_context(_my_style):
        yield


def my_style(func):
    """
    Decorator to use my home-made plt style within a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_my_style):
            return func(*args, **kwargs)

    return wrapper


def move_axes(ax: plt.Axes | None = None):  # type: ignore
    """
    Move the axes to the center of the figure
    """
    ax: plt.Axes = ax or plt.gca()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


@my_style
def parity_plot(
    model: GraphPESModel,
    graphs: AtomicGraphBatch | list[AtomicGraph],
    property: PropertyKey = Property.ENERGY,
    property_label: str | None = None,
    transform: Transform | None = None,
    units: str | None = None,
    ax: plt.Axes | None = None,  # type: ignore
    **scatter_kwargs,
):
    r"""
    A nicely formatted parity plot of model predictions vs ground truth
    for the given :code:`property`.

    Parameters
    ----------
    model
        The model to for generating predictions.
    graphs
        The graphs to make predictions on.
    property
        The property to plot, e.g. :code:`Property.ENERGY`.
    property_label
        The string that the property is indexed by on the graphs. If not
        provided, defaults to the value of :code:`property`, e.g.
        :code:`Property.ENERGY` :math:`\rightarrow` :code:`"energy"`.
    transform
        The transform to apply to the predictions and labels before plotting.
        If not provided, no transform is applied.
    units
        The units of the property, for labelling the axes. If not provided, no
        units are used.
    ax
        The axes to plot on. If not provided, the current axes are used.
    scatter_kwargs
        Keyword arguments to pass to :code:`plt.scatter`.

    Examples
    --------
    Default settings (no units, transforms or custom scatter keywords):

    .. code-block:: python

        parity_plot(model, train, Property.ENERGY)

    .. image:: notebooks/Cu-LJ-default-parity.svg
        :align: center

    Custom settings, as seen in
    :doc:`this example notebook <notebooks/example>`:

    .. code-block:: python

        from graph_pes.transform import DividePerAtom
        from graph_pes.util import Keys

        for name, data, colour in zip(
            ["Train", "Test"],
            [train, test],
            ["royalblue", "crimson"],
        ):
            parity_plot(
                model,
                data,
                Property.ENERGY,
                transform=DividePerAtom(),
                units="eV / atom",
                label=name,
                c=colour,
            )

        plt.legend(loc="upper left", fancybox=False);

    .. image:: notebooks/Cu-LJ-parity.svg
        :align: center
    """
    # deal with defaults
    transform = transform or Identity()
    if property_label is None:
        property_label = property

    # get the predictions and labels
    if isinstance(graphs, list):
        graphs = AtomicGraphBatch.from_graphs(graphs)

    ground_truth = transform(graphs[property_label], graphs).detach()
    predictions = transform(
        model.predict(graphs, property=property), graphs
    ).detach()

    # plot
    ax: plt.Axes = ax or plt.gca()

    default_kwargs = dict(lw=0, clip_on=False)
    scatter_kwargs = {**default_kwargs, **scatter_kwargs}
    ax.scatter(ground_truth, predictions, **scatter_kwargs)

    # get a point guaranteed to be on the plot
    z = ground_truth.view(-1)[0].item()
    ax.axline((z, z), slope=1, c="k", ls="--", lw=1)

    # aesthetics
    axis_label = f"{property_label}  ({units})" if units else property_label
    ax.set_xlabel(f"True {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_aspect("equal", "datalim")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, y0), max(x1, y1))
    ax.set_ylim(min(x0, y0), max(x1, y1))
    move_axes(ax)

    # 5 ticks each
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


@my_style
def dimer_curve(
    model: GraphPESModel,
    system: str,
    units: str | None = None,
    set_to_zero: bool = True,
    rmin: float = 0.9,
    rmax: float = 5.0,
    ax: plt.Axes | None = None,  # type: ignore
    **plot_kwargs,
):
    r"""
    A nicely formatted dimer curve plot for the given :code:`system`.

    Parameters
    ----------
    model
        The model to for generating predictions.
    system
        The dimer system. Should be one of: a single element, e.g. :code:`"Cu"`,
        or a pair of elements, e.g. :code:`"CuO"`.
    units
        The units of the energy, for labelling the axes. If not provided, no
        units are used.
    set_to_zero
        Whether to set the energy of the dimer at :code:`rmax` to be zero.
    rmin
        The minimum seperation to consider.
    rmax
        The maximum seperation to consider.
    ax
        The axes to plot on. If not provided, the current axes are used.
    plot_kwargs
        Keyword arguments to pass to :code:`plt.plot`.

    Examples
    --------
    See :doc:`this example notebook <notebooks/example>`:

    .. code-block:: python

        dimer_curve(model, "Cu", units="eV", label="Final", c="C1")

    .. image:: notebooks/Cu-LJ-dimer.svg
        :align: center
    """

    trial_atoms = Atoms(system)
    if len(trial_atoms) != 2:
        system = system + "2"

    rs = np.linspace(rmin, rmax, 200)
    atoms = [Atoms(system, positions=[[0, 0, 0], [r, 0, 0]]) for r in rs]
    graphs = convert_to_atomic_graphs(atoms, cutoff=rmax + 0.1)
    batch = AtomicGraphBatch.from_graphs(graphs)

    with torch.no_grad():
        energy = model(batch).numpy()
    if set_to_zero:
        energy -= energy[-1]

    ax: plt.Axes = ax or plt.gca()

    default_kwargs = dict(lw=1, c="k")
    plot_kwargs = {**default_kwargs, **plot_kwargs}
    ax.plot(rs, energy, **plot_kwargs)

    limiting_energy = energy[-1]
    if (energy[:-1] < limiting_energy).any():
        well_depth = limiting_energy - energy[:-1].min()
    else:
        well_depth = 0.1
    bottom = limiting_energy - well_depth * 1.1
    top = limiting_energy + well_depth * 1.1
    ax.set_ylim(bottom, top)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel(f"Dimer Energy ({units})" if units else "Dimer Energy")

    first_in_view = np.where(energy < top)[0][0]
    ax.set_xlim(rs[first_in_view].item() - 0.2, rs[-1] + 0.2)
    move_axes(ax)

    # 5 ticks each
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
