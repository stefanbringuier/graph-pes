from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, NamedTuple, Sequence

import torch
import torchmetrics
from torch import Tensor, nn
from torchmetrics import Metric as TorchMetric

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    divide_per_atom,
    is_batch,
    number_of_structures,
    replace,
    sum_per_structure,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.misc import differentiate, uniform_repr
from graph_pes.utils.nn import UniformModuleList

Metric = Callable[[Tensor, Tensor], Tensor]
MetricName = Literal["MAE", "RMSE", "MSE"]


class WeightedLoss:
    def __init__(self, *args, **kwargs):
        # this is now depracated: pass weight directly to the loss object
        raise ImportError(
            "The WeightedLoss class has been removed from graph-pes "
            "as of version 0.0.22. Please now pass loss weights directly "
            "to the loss instances! See the docs for more information: "
            "https://jla-gardner.github.io/graph-pes/fitting/losses.html"
        )


class Loss(nn.Module, ABC):
    """
    A general base class for all loss functions in ``graph-pes``.

    Implementations **must** override:

    * :meth:`forward` to compute the loss value.
    * :meth:`name` to return the name of the loss function.
    * :meth:`required_properties` to return the properties that this loss
      function needs to have available in order to compute its value.

    Additionally, implementations can optionally override:

    * :meth:`pre_fit` to perform any necessary operations before training
      commences.

    Parameters
    ----------
    weight
        a scalar multiplier for weighting the value returned by
        :meth:`forward` as part of a :class:`TotalLoss`.
    is_per_atom
        whether this loss returns a value that is normalised per atom, or not.
        For instance, some metric that acts on ``"forces"`` is naturally
        per-atom, while a metric that acts on ``"energy"``, or the model etc.,
        is not. Specifying this correctly ensures that the effective batch size
        is chosen correctly when averaging over batches.
    """

    def __init__(self, weight, is_per_atom: bool = False):
        super().__init__()
        self.weight = weight
        self.is_per_atom = is_per_atom

    @abstractmethod
    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        r"""
        Compute the unweighted loss value.

        Note that only :class:`Loss`\ s that return a tensor can be used
        for training: we reserve the use of :class:`TorchMetric`\ s for
        evaluation metrics only.

        :class:`Loss`\ s can act on any of:

        Parameters
        ----------
        model
            The model being trained.
        graph
            The graph (usually a batch) the ``model`` was applied to.
        predictions
            The predictions from the ``model`` for the given ``graph``.
        """

    @property
    @abstractmethod
    def required_properties(self) -> list[PropertyKey]:
        """The properties that are required by this loss function."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this loss function, for logging purposes."""

    def pre_fit(self, training_data: AtomicGraph):
        """
        Perform any necessary operations before training commences.

        For example, this could be used to pre-compute a standard deviation
        of some property in the training data, which could then be used in
        :meth:`forward`.

        Parameters
        ----------
        training_data
            The training data to pre-fit this loss function to.
        """

    # add type hints to play nicely with mypy
    def __call__(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        return super().__call__(model, graph, predictions)


class PropertyLoss(Loss):
    r"""
    A :class:`PropertyLoss` instance applies its :class:`Metric` to compare a
    model's predictions to the true values for a given property of a
    :class:`~graph_pes.AtomicGraph`.

    Parameters
    ----------
    property
        The property to apply the loss metric to.
    metric
        The loss metric to use. Defaults to :class:`RMSE`.

    Examples
    --------

    .. code-block:: python

        energy_rmse_loss = PropertyLoss("energy", RMSE())
        energy_rmse_value = energy_rmse_loss(
            predictions,  # a dict of key (energy/force/etc.) to value
            graph.properties,
        )
    """

    def __init__(
        self,
        property: PropertyKey,
        metric: Metric | MetricName | TorchMetric = "RMSE",
        weight: float = 1.0,
    ):
        super().__init__(
            weight,
            is_per_atom=property in ("forces", "local_energies"),
        )
        self.property: PropertyKey = property
        self.metric = parse_metric(metric)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        """
        Computes the loss value.

        Parameters
        ----------
        predictions
            The predictions from the ``model`` for the given ``graph``.
        """

        if isinstance(self.metric, torchmetrics.Metric):
            self.metric.update(
                predictions[self.property],
                graph.properties[self.property],
            )
            return self.metric

        return self.metric(
            predictions[self.property],
            graph.properties[self.property],
        )

    @property
    def name(self) -> str:
        """Get the name of this loss for logging purposes."""
        return f"{self.property}_{_get_metric_name(self.metric)}"

    @property
    def required_properties(self) -> list[PropertyKey]:
        return [self.property]

    def __repr__(self) -> str:
        return uniform_repr(
            self.__class__.__name__,
            self.property,
            metric=self.metric,
        )


class SubLossPair(NamedTuple):
    loss_value: torch.Tensor
    weighted_loss_value: torch.Tensor
    is_per_atom: bool


class TotalLossResult(NamedTuple):
    loss_value: torch.Tensor
    components: dict[str, SubLossPair]


class TotalLoss(torch.nn.Module):
    r"""
    A lightweight wrapper around a collection of losses.

    .. math::

        \mathcal{L}_{\text{total}} = \sum_i w_i \mathcal{L}_i

    where :math:`\mathcal{L}_i` is the :math:`i`-th loss and :math:`w_i` is the
    corresponding weight.

    ``graph-pes`` models are trained by minimising a :class:`TotalLoss` value.

    Parameters
    ----------
    losses
        The collection of losses to aggregate.
    """

    def __init__(self, losses: Sequence[Loss]):
        super().__init__()
        self.losses = UniformModuleList(losses)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> TotalLossResult:
        """
        Computes the total loss value.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graph
            The graph (usually a batch) the ``model`` was applied to.
        """

        total_loss = torch.scalar_tensor(0.0, device=graph.Z.device)
        components: dict[str, SubLossPair] = {}

        for loss in self.losses:
            loss_value = loss(model, graph, predictions)
            weighted_loss_value = loss_value * loss.weight

            if not isinstance(loss_value, torch.Tensor):
                raise ValueError(
                    f"Total losses can only be used alongside metrics that "
                    f"return a tensor. Instead, loss {loss.name} returned "
                    f"{type(loss_value)}."
                )

            total_loss += weighted_loss_value
            components[loss.name] = SubLossPair(
                loss_value,
                weighted_loss_value,
                loss.is_per_atom,
            )

        return TotalLossResult(total_loss, components)

    # add type hints to appease mypy
    def __call__(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> TotalLossResult:
        return super().__call__(model, graph, predictions)

    def __repr__(self) -> str:
        return "\n".join(
            ["TotalLoss:"]
            + ["    ".join(str(loss).split("\n")) for loss in self.losses]
        )


############################# CUSTOM LOSSES #############################


class PerAtomEnergyLoss(PropertyLoss):
    r"""
    A loss function that evaluates some metric on the total energy normalised
    by the number of atoms in the structure.

    .. math::
        \mathcal{L} = \text{metric}\left(
        \bigoplus_i \frac{\hat{E}_i}{N_i}, \bigoplus_i\frac{E_i}{N_i} \right)

    where :math:`\hat{E}_i` is the predicted energy for structure :math:`i`,
    :math:`E_i` is the true energy for structure :math:`i`, :math:`N_i`
    is the number of atoms in structure :math:`i` and :math:`\bigoplus_i`
    denotes the cocatenation over all structures in the batch.

    Parameters
    ----------
    metric
        The loss metric to use. Defaults to :class:`RMSE`.
    """

    def __init__(
        self,
        metric: Metric | MetricName | TorchMetric = "RMSE",
        weight: float = 1.0,
    ):
        super().__init__("energy", metric, weight)

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor | TorchMetric:
        if isinstance(self.metric, torchmetrics.Metric):
            self.metric.update(
                divide_per_atom(predictions["energy"], graph),
                divide_per_atom(graph.properties["energy"], graph),
            )
            return self.metric

        return self.metric(
            divide_per_atom(predictions["energy"], graph),
            divide_per_atom(graph.properties["energy"], graph),
        )

    @property
    def name(self) -> str:
        return f"per_atom_energy_{_get_metric_name(self.metric)}"


class ForceRMSE(PropertyLoss):
    """
    Alias for :class:`PropertyLoss` with ``property="forces"`` and
    ``metric=RMSE``.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("forces", RMSE(), weight)


class EquigradLoss(Loss):
    r"""Loss that penalizes violations of rotational equivariance.

    This loss is based on the original Orb (orb-v3) implementation
    as described in [Orbv3]_. The energy of a system should be independent
    of global rotation. To achieve this, this loss computes rotational
    gradients, i.e., the gradients of energy :math:`E` with respect to
    rotations
    :math:`\theta_x, \theta_y, \theta_z` around the x, y, and z axes:
    :math:`\nabla_\theta E = (\partial E/\partial \theta_x,
    \partial E/\partial \theta_y, \partial E/\partial \theta_z)`.

    For a perfectly equivariant model, these gradients should be zero:
    :math:`\nabla_\theta E = \mathbf{0}`. Non-zero gradients indicate that the
    model's energy predictions depend on the absolute orientation of
    the system, violating physical symmetry.

    The loss uses a finite difference approach:

    1. Compute the energy of the system in its original orientation
    2. Apply small rotations around each coordinate axis
    3. Compute the energy after each rotation
    4. Calculate the change in energy divided by the rotation angle:
       :math:`\partial E/\partial \theta_i \approx
       (E(\theta_i) - E(0))/\theta_i`
    5. Take the RMS norm of these rotational gradients as the loss value:
       :math:`\mathcal{L}_{\text{equigrad}} = \|\nabla_\theta E\|_2`

    .. note::
        This implementation differs from the original Orb (orb-v3)
        implementation [Orbv3]_ in two ways:

        1. Orb v3 uses autograd to compute exact gradients of energy with
        respect to rotational DOF, whereas this implementation uses a
        finite-difference approach. This makes the implementation more general
        and compatible with any `graph-pes` model that can compute energies,
        but potentially less accurate than the autograd approach.

        The forward finite-difference method :math:`(f(x+h) - f(x))/h` is used
        for computational efficiency, requiring only one additional
        forward pass per rotation axis. This assumption should be valid
        when using small rotation angles. Proceed caustiously for larger
        rotation angles (:math:`\gg 0.01` radians) should be avoided as
        they will likely lead to poor gradient estimates and/or numerical
        instability during training. Extremely small angles (:math:`\ll 0.001`)
        might suffer from floating-point precision issues.

        2. For constructing rotation matrices, Orb v3 uses matrix exponentials
        of skew-symmetric generators: :math:`R = \exp(S - S^T)` where :math:`S`
        is the generator. Our implementation directly constructs rotation
        matrices using sine/cosine functions for each rotation axis since
        we should restrict ourselves to small angles given point 1. This
        should produce mathematically equivalent rotation matrices and is
        likely more computationally efficient with negligible difference.



    Parameters
    ----------
    weight : float
        The weight of this loss in the total loss function.
    rotation_mag : float, default=0.01
        The *small* rotation angle (in radians) used for gradient calculations.
        Default: 0.01 radians (approximately 0.6 degrees).
    method_type : str, default="forward"
        Finite difference method to use: "forward" or "central".

    Examples
    --------

    .. code-block:: yaml

        loss:
          +EquigradLoss:
              weight: 0.1
              rotation_mag: 0.01  # radians
              method_type: "central"

    References
    ----------
    .. [Orbv3] B. Rhodes, S. Vandenhaute, V. Simkus, J. Gin, J. Godwin,
              T. Duignan, M. Neumann, "Orb-v3: atomistic simulation at scale",
              (2025). https://arxiv.org/abs/2504.06231
    """

    def __init__(
        self,
        weight: float = 1.0,
        rotation_mag: float = 0.01,
        method_type: str = "forward",
    ):
        """
        Initialize the EquigradLoss.

        Parameters
        ----------
        weight : float
            Weight of this loss component in the total loss
        rotation_mag : float
            Magnitude of rotation in radians used for finite difference calc.
        method_type : str
            Finite difference method to use: "forward" or "central"
        """
        super().__init__(weight=weight, is_per_atom=False)
        self.rotation_mag = rotation_mag

        if method_type not in ["forward", "central"]:
            raise ValueError(
                f"method_type must be 'forward' or 'central', got {method_type}"
            )
        self.method_type = method_type

    def compute_rotational_gradients(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
    ) -> torch.Tensor:
        """
        Compute rotational gradients of the energy with respect to rotation.


        Parameters
        ----------
        model
            The model to compute gradients for
        graph
            The graph to use for computation

        Returns
        -------
        torch.Tensor
            Rotational gradients tensor of shape (n_structures, 3, 1)
        """
        device = graph.Z.device
        batch_size = 1 if not is_batch(graph) else number_of_structures(graph)

        rot_grads = torch.zeros(batch_size, 3, 1, device=device)

        # Radians
        delta = self.rotation_mag

        # Forward pass, no rotation
        with torch.no_grad():
            output = model(graph)

            if "energy" in output:
                base_energy = output["energy"]
            elif "local_energies" in output:
                base_energy = sum_per_structure(output["local_energies"], graph)
            else:
                return rot_grads

        # Cell axes (x, y, z)
        for axis in range(3):
            if self.method_type == "forward":
                rot_matrix = torch.eye(3, device=device)

                # Modify appropriate cell axes
                if axis == 0:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    rot_matrix[1, 1], rot_matrix[1, 2] = c, -s
                    rot_matrix[2, 1], rot_matrix[2, 2] = s, c
                elif axis == 1:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    rot_matrix[0, 0], rot_matrix[0, 2] = c, s
                    rot_matrix[2, 0], rot_matrix[2, 2] = -s, c
                else:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    rot_matrix[0, 0], rot_matrix[0, 1] = c, -s
                    rot_matrix[1, 0], rot_matrix[1, 1] = s, c

                # Center about COM
                if is_batch(graph):
                    centers = torch.zeros((batch_size, 3), device=device)
                    for i in range(batch_size):
                        mask = graph.batch == i
                        centers[i] = graph.R[mask].mean(dim=0)
                    # Map centers
                    expanded_centers = centers[graph.batch]
                else:
                    center = graph.R.mean(dim=0)
                    expanded_centers = center.expand_as(graph.R)

                centered_R = graph.R - expanded_centers
                rotated_R = torch.matmul(centered_R, rot_matrix.t()) + expanded_centers

                rotated_graph = replace(graph, R=rotated_R)

                # Foward pass w/ rotation
                with torch.no_grad():
                    rot_output = model(rotated_graph)

                    if "energy" in rot_output:
                        rot_energy = rot_output["energy"]
                    elif "local_energies" in rot_output:
                        rot_energy = sum_per_structure(
                            rot_output["local_energies"], rotated_graph
                        )
                    else:
                        # Can't compute energy?
                        continue

                rot_grad = (rot_energy - base_energy) / delta

                # Store axes gradient
                for i in range(batch_size):
                    if is_batch(graph):
                        # Check if rot_grad is a scalar (0-dim tensor)
                        if rot_grad.ndim == 0:
                            rot_grads[i, axis, 0] = rot_grad
                        else:
                            rot_grads[i, axis, 0] = rot_grad[i]
                    else:
                        rot_grads[i, axis, 0] = rot_grad

            # Central difference method
            # Same general steps as above!
            else:
                pos_rot_matrix = torch.eye(3, device=device)
                neg_rot_matrix = torch.eye(3, device=device)

                if axis == 0:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    pos_rot_matrix[1, 1], pos_rot_matrix[1, 2] = c, -s
                    pos_rot_matrix[2, 1], pos_rot_matrix[2, 2] = s, c
                    c, s = torch.cos(torch.tensor(-delta)), torch.sin(
                        torch.tensor(-delta)
                    )
                    neg_rot_matrix[1, 1], neg_rot_matrix[1, 2] = c, -s
                    neg_rot_matrix[2, 1], neg_rot_matrix[2, 2] = s, c
                elif axis == 1:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    pos_rot_matrix[0, 0], pos_rot_matrix[0, 2] = c, s
                    pos_rot_matrix[2, 0], pos_rot_matrix[2, 2] = -s, c
                    c, s = torch.cos(torch.tensor(-delta)), torch.sin(
                        torch.tensor(-delta)
                    )
                    neg_rot_matrix[0, 0], neg_rot_matrix[0, 2] = c, s
                    neg_rot_matrix[2, 0], neg_rot_matrix[2, 2] = -s, c
                else:
                    c, s = torch.cos(torch.tensor(delta)), torch.sin(
                        torch.tensor(delta)
                    )
                    pos_rot_matrix[0, 0], pos_rot_matrix[0, 1] = c, -s
                    pos_rot_matrix[1, 0], pos_rot_matrix[1, 1] = s, c
                    c, s = torch.cos(torch.tensor(-delta)), torch.sin(
                        torch.tensor(-delta)
                    )
                    neg_rot_matrix[0, 0], neg_rot_matrix[0, 1] = c, -s
                    neg_rot_matrix[1, 0], neg_rot_matrix[1, 1] = s, c

                if is_batch(graph):
                    centers = torch.zeros((batch_size, 3), device=device)
                    for i in range(batch_size):
                        mask = graph.batch == i
                        centers[i] = graph.R[mask].mean(dim=0)
                    expanded_centers = centers[graph.batch]
                else:
                    center = graph.R.mean(dim=0)
                    expanded_centers = center.expand_as(graph.R)

                centered_R = graph.R - expanded_centers
                pos_rotated_R = (
                    torch.matmul(centered_R, pos_rot_matrix.t()) + expanded_centers
                )
                pos_rotated_graph = replace(graph, R=pos_rotated_R)

                neg_rotated_R = (
                    torch.matmul(centered_R, neg_rot_matrix.t()) + expanded_centers
                )
                neg_rotated_graph = replace(graph, R=neg_rotated_R)

                with torch.no_grad():
                    pos_rot_output = model(pos_rotated_graph)
                    neg_rot_output = model(neg_rotated_graph)

                    # Positive rotation energy
                    if "energy" in pos_rot_output:
                        pos_rot_energy = pos_rot_output["energy"]
                    elif "local_energies" in pos_rot_output:
                        pos_rot_energy = sum_per_structure(
                            pos_rot_output["local_energies"], pos_rotated_graph
                        )
                    else:
                        continue

                    # Negative rotation energy
                    if "energy" in neg_rot_output:
                        neg_rot_energy = neg_rot_output["energy"]
                    elif "local_energies" in neg_rot_output:
                        neg_rot_energy = sum_per_structure(
                            neg_rot_output["local_energies"], neg_rotated_graph
                        )
                    else:
                        continue

                rot_grad = (pos_rot_energy - neg_rot_energy) / (2 * delta)

                for i in range(batch_size):
                    if is_batch(graph):
                        # Check if rot_grad is a scalar (0-dim tensor)
                        if rot_grad.ndim == 0:
                            rot_grads[i, axis, 0] = rot_grad
                        else:
                            rot_grads[i, axis, 0] = rot_grad[i]
                    else:
                        rot_grads[i, axis, 0] = rot_grad

        return rot_grads

    def forward(
        self,
        model: GraphPESModel,
        graph: AtomicGraph,
        predictions: dict[PropertyKey, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the equigrad loss using RMS metric.

        Parameters
        ----------
        model : GraphPESModel
            The model being trained
        graph : AtomicGraph
            The atomic graph to compute the loss for
        predictions : dict[PropertyKey, torch.Tensor]
            (Not used) The model's predictions for the graph

        Returns
        -------
        torch.Tensor
            The equigrad loss value
        """

        rotational_grad = self.compute_rotational_gradients(model, graph)

        # Calculate RMS of the gradients
        equigrad_rms = torch.norm(rotational_grad, dim=(1, 2)).mean()

        return self.weight * equigrad_rms

    @property
    def required_properties(self) -> list[PropertyKey]:
        """This loss relies on the energy predictions, which are calculated internally
        in forward."""
        return ["energy"]

    @property
    def name(self) -> str:
        """Get the name of this loss for logging purposes."""
        return "equigrad"


## METRICS ##


def parse_metric(metric: Metric | MetricName | TorchMetric | None) -> Metric:
    if isinstance(metric, str):
        return {"MAE": MAE(), "RMSE": RMSE(), "MSE": MSE()}[metric]

    if metric is None:
        return RMSE()

    return metric


class MSE(torch.nn.MSELoss):
    r"""
    Mean squared error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2
    """


class RMSE(torch.nn.MSELoss):
    r"""
    Root mean squared error metric:

    .. math::
        \sqrt{ \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2 }

    .. note::

        Metrics are computed per-batch in the Lightning Trainer.
        When aggregating this metric across multiple batches, we therefore
        get the mean of the per-batch RMSEs, which is not the same as the
        RMSE between all predictions and targets from all batches.

        ``graph-pes`` avoids this issue for all RMSE-based metrics logged as
        ``"{valid|test}/metrics/..._rmse"`` by using a different,
        ``torchmetrics`` based implementation.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (super().forward(input, target)).sqrt()

    @property
    def name(self) -> str:
        return "rmse_batchwise"


class MAE(torch.nn.L1Loss):
    r"""
    Mean absolute error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left| \hat{P}_i - P_i \right|
    """


class Huber(torch.nn.HuberLoss):
    r"""
    Huber loss metric:

    .. math::
        L_\delta(y, \hat{y}) = \begin{cases}
            \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
            \delta (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
        \end{cases}
    """

    def __init__(self, delta: float = 0.5):
        super().__init__(delta=delta)

    @property
    def name(self) -> str:
        return f"huber_{self.delta}"


class ScaleFreeHuber(Huber):
    r"""
    A scaled version of the :class:`Huber` loss metric, such that,
    beyond :math:`|y - \hat{y}| = \delta`, the loss is independent of
    :math:`\delta`: this makes tuning the :math:`\delta` value
    easier when several loss components are being used (no need to also
    change the relative weighting of the loss components).

    .. math::
        \mathcal{L}_{\delta}(y, \hat{y}) =
        \frac{1}{\delta} \text{Huber}_\delta(y, \hat{y})
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target) / self.delta

    @property
    def name(self) -> str:
        return f"scaled_huber_{self.delta}"


def _get_metric_name(metric: Metric) -> str:
    if hasattr(metric, "name"):
        return metric.name  # type: ignore

    if isinstance(metric, torchmetrics.MeanSquaredError):
        return "mse" if metric.squared else "rmse"

    # if metric is a function, we want the function's name, otherwise
    # we want the metric's class name, all lowercased
    # and without the word "loss" in it
    return (
        getattr(
            metric,
            "__name__",
            metric.__class__.__name__,
        )
        .lower()
        .replace("loss", "")
    )
