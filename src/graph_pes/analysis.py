from __future__ import annotations

from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .core import GraphPESModel, get_predictions
from .data.atomic_graph import AtomicGraph
from .data.batching import AtomicGraphBatch
from .transform import Identity, Transform
from .util import Keys


def my_style(func):
    """
    Decorator to use my home-made plt style within a function
    """

    style = {
        "figure.figsize": (3, 3),
        "axes.spines.right": False,
        "axes.spines.top": False,
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(style):
            return func(*args, **kwargs)

    return wrapper


def move_axes(ax):
    """
    Move the axes to the center of the figure
    """
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


@my_style
def parity_plot(
    model: GraphPESModel,
    graphs: AtomicGraphBatch | list[AtomicGraph],
    property: Keys,
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
        The property to plot, e.g. :code:`Keys.ENERGY`.
    property_label
        The string that the property is indexed by on the graphs. If not
        provided, defaults to the value of :code:`property`, e.g.
        :code:`Keys.ENERGY` :math:`\rightarrow` :code:`"energy"`.
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

        parity_plot(model, train, Keys.ENERGY)

    .. image:: notebooks/Cu-LJ-default-parity.svg
        :align: center

    Custom settings, as seen in
    :doc:`this example notebook <notebooks/example>`:

    .. code-block:: python

        from graph_pes.transform import DividePerAtom
        from graph_pes.util import Keys

        parity_plot(
            model,
            train,
            Keys.ENERGY,
            transform=DividePerAtom(),
            units="eV/atom",
            c="royalblue",
            label="Train",
        )

        ...

    .. image:: notebooks/Cu-LJ-parity.svg
        :align: center
    """
    # deal with defaults
    transform = transform or Identity()
    if property_label is None:
        property_label = property.value

    # get the predictions and labels
    if isinstance(graphs, list):
        graphs = AtomicGraphBatch.from_graphs(graphs)

    ground_truth = transform(graphs[property_label], graphs).detach()
    predictions = transform(
        get_predictions(model, graphs, {property: property_label})[
            property_label
        ],
        graphs,
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
