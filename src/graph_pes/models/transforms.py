from __future__ import annotations

import torch
from torch import nn

from graph_pes.data import AtomicGraph
from graph_pes.data.batching import (
    AtomicGraphBatch,
    sum_per_structure,
)
from graph_pes.util import MAX_Z, to_chem_symbol


from abc import ABC, abstractmethod


class LocalEnergyTransform(nn.Module):
    r"""
    An optionally learnable per-species transformation of local energies:

    .. math::

        T : \varepsilon_z \mapsto \varepsilon_z \cdot \sigma_z + \mu_z
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable
        self.mu = nn.Parameter(torch.zeros(MAX_Z + 1), trainable)
        self.sigma = nn.Parameter(torch.ones(MAX_Z + 1), trainable)

    def forward(
        self, local_energies: torch.Tensor, Zs: torch.Tensor
    ) -> torch.Tensor:
        return self.mu[Zs] + self.sigma[Zs] * local_energies

    def fit_to(self, graphs: list[AtomicGraph]):
        # tricky
        # for now, just do best guess based on linear reg
        batch = AtomicGraphBatch.from_graphs(graphs)
        total_energy = batch.labels["energy"]

        # get the matrix `N` where N[i,j] is the number of atoms of species j
        # in structure i
        N = torch.zeros(len(graphs), MAX_Z + 1)
        for i, g in enumerate(graphs):
            for j, z in enumerate(torch.unique(g.Z)):
                N[i, z] = (g.Z == z).sum()

        offsets = torch.linalg.lstsq(N, total_energy).solution
        self.mu = nn.Parameter(offsets, self.trainable)

        # now to guess the scales
        # for now we just get the std and assign it to all species
        mean_energies = (N @ offsets).squeeze()
        std = (total_energy - mean_energies).std()
        sigma = torch.ones(MAX_Z + 1) * std
        self.sigma = nn.Parameter(sigma, self.trainable)

    def __repr__(self):
        # only care about the relevant offsets and scales
        # we do this by finding the indices of the non-zero values
        # and then only printing those
        relevant = torch.nonzero(self.mu.detach() != 0)
        offsets = {
            to_chem_symbol(int(z.item())): round(self.mu[z].item(), 2)
            for z in relevant
        }
        scales = {
            to_chem_symbol(int(z.item())): round(self.sigma[z].item(), 2)
            for z in relevant
        }

        return (
            f"{self.__class__.__name__}(\n"
            f"  offsets={offsets},\n  scales={scales}\n)"
        )


class PreLossNormaliser(nn.Module, ABC):
    """
    Responsible for normalising predictions and labels
    before they are passed to the loss function.
    """

    @abstractmethod
    def fit_to(self, graphs: list[AtomicGraph]):
        """fit the normaliser to the provided data"""

    @abstractmethod
    def forward(
        self, thing: torch.Tensor, graph: AtomicGraph | AtomicGraphBatch
    ):
        """normalise the thing, given the graph"""


class EnergyNormaliser(PreLossNormaliser):
    r"""
    Implements a normalisation, T, of total energies as a function
    of the number of atoms present in the structure:

    .. math::

        T : E \mapsto \frac{E}{\sqrt{N}}

    where :math:`N` is the number of atoms in the structure.
    """

    def forward(
        self, total_energy: torch.Tensor, graph: AtomicGraph | AtomicGraphBatch
    ) -> torch.Tensor:
        if isinstance(graph, AtomicGraphBatch):
            atoms_per_structure = graph.ptr[1:] - graph.ptr[:-1]
        else:
            atoms_per_structure = torch.scalar_tensor(graph.n_atoms)

        return total_energy / atoms_per_structure.sqrt()

    def fit_to(self, graphs: list[AtomicGraph]):
        pass


class ForceNormaliser(PreLossNormaliser):
    r"""
    Implements a per-species normalisation, T, of forces, dividing by the
    standard deviation of the force magnitude for that species:

    .. math::

        T : \mathbf{F_i} \mapsto \frac{\mathbf{F_i}}{\eta_{Z_i}}
    """

    def __init__(self):
        super().__init__()
        self.eta = torch.ones(1, MAX_Z + 1)

    def forward(
        self, forces: torch.Tensor, graph: AtomicGraph | AtomicGraphBatch
    ) -> torch.Tensor:
        return forces / self.eta[graph.Z].unsqueeze(-1)

    def fit_to(self, graphs: list[AtomicGraph]):
        batch = AtomicGraphBatch.from_graphs(graphs)
        force_mags = batch.labels["forces"].norm(dim=-1)
        Zs = batch.Z

        # get a stddev of force magnitude per species
        eta = torch.ones(MAX_Z + 1)
        for Z in torch.unique(Zs):
            eta[Z] = force_mags[Zs == Z].std()

        self.eta = eta


class Transforms(nn.Module, ABC):
    r"""
    Implementations of this class are used to transform total system energy
    and forces into a standardized space and back again, via the
    :meth:`standardize` and :meth:`unstandardize` methods.
    """

    @abstractmethod
    def standardize(
        self,
        total_energy: torch.Tensor,
        forces: torch.Tensor,
        graphs: AtomicGraph | AtomicGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize the force and total energy labels, given the graph/s.

        Parameters
        ----------
        total_energy : torch.Tensor
            The total energy to standardize.
        forces : torch.Tensor
            The forces to standardize.
        graphs : AtomicGraph | AtomicGraphBatch
            The graph/s containing the atoms.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The standardized total energy and forces.
        """


class Standardizer(nn.Module):
    r"""
    Transforms for standardizing the total energy and forces on a structure.

    Assume that the total energy is a sum of local energies, each of which
    is normally distributed (in the absence of any other information):

    .. math::
        
        \begin{align*}
            E &= \sum_{i} \varepsilon_{Z_i} \\
            \varepsilon_{Z_i} &\sim \mathcal{N}(\mu_{Z_i}, \sigma_{Z_i}^2)
        \end{align*}

    Then the total energy of a structure is also normally distributed:

    .. math::

        E \sim \mathcal{N}(\mu, \sigma^2)

    where :math:`\mu = \sum_{i} \mu_{Z_i}` and 
    :math:`\sigma^2 = \sum_{i} \sigma_{Z_i}^2`.

    The standardizing transformation is thus:

    .. math::

        E_{\text{std}} = T(E, Z) = 
        \frac{E - \sum_{i} \mu_{Z_i}}{\sqrt{\sum_{i} \sigma_{Z_i}^2}}

    And the inverse transformation, for local energies, is:

    .. math::

        \varepsilon_{i} = T^{-1}(\varepsilon_{\text{std},i}, Z_i) = 
        \mu_{Z_i} + \sigma_{Z_i} \varepsilon_{\text{std},i}


    Given that the forces are the negative gradient of the total energy,
    :math:`\mathbf{F} = -\nabla E`, we get 
    :math:`\mathbf{F}_{\text{std},i} = \mathbf{F}_{i} / \sigma_{Z_i}`.
    In general, however, this does not mean that the resulting, 
    "standardized" forces have unit variance. We therefore also scale these
    by $\eta_{Z_i}$, such that :math:`\| \mathbf{F}_{\text{std}} \|` are
    half-normally distributed with unit variance.
        
    """

    def __init__(self, shared_energy_scale: bool = False):
        super().__init__()
        self.energy_offset = torch.zeros(MAX_Z + 1)

        self.shared_energy_scale = shared_energy_scale
        if shared_energy_scale:
            self.energy_scale = torch.scalar_tensor(1)
        else:
            self.energy_scale = torch.ones(MAX_Z + 1)

        self.force_eta = torch.ones(MAX_Z + 1)

    def standardize(
        self,
        total_energy: torch.Tensor,
        forces: torch.Tensor,
        graphs: AtomicGraph | AtomicGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize the force and total energy labels, given the graph/s.

        Parameters
        ----------
        total_energy : torch.Tensor
            The total energy to standardize.
        forces : torch.Tensor
            The forces to standardize.
        graphs : AtomicGraph | AtomicGraphBatch
            The graph/s containing the atoms.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The standardized total energy and forces.
        """

        means = self.energy_offset[graphs.Z]
        if self.shared_energy_scale:
            sigmas = torch.ones_like(graphs.Z) * self.energy_scale
        else:
            sigmas = self.energy_scale[graphs.Z]

        total_mean = sum_per_structure(means, graphs)
        total_sigma = sum_per_structure(sigmas.pow(2), graphs).sqrt()

        return (
            # energy is per structure, and hence scaled by total statistics
            (total_energy - total_mean) / total_sigma,
            # forces are per atom, and hence scaled by per-species statistics
            forces / sigmas.unsqueeze(-1),
        )

    def unstandardize(
        self,
        local_energies: torch.Tensor,
        graphs: AtomicGraph | AtomicGraphBatch,
    ) -> torch.Tensor:
        """
        Unstandardize the local energies, given the graph.

        Parameters
        ----------
        local_energies : torch.Tensor
            The local energies to unstandardize.
        graphs : AtomicGraph | AtomicGraphBatch
            The graph/s containing the atoms.

        Returns
        -------
        torch.Tensor
            The unstandardized local energies.
        """

        if self.shared_energy_scale:
            sigma = self.energy_scale
        else:
            sigma = self.energy_scale[graphs.Z]

        return self.energy_offset[graphs.Z] + sigma * local_energies

    def fit_to(self, graphs: list[AtomicGraph]):
        # convert graphs to the following format:
        # num_atoms: a 2D tensor of shape (num_structures, num_distinc_species)
        #   where num_atoms[i,j] is the number of atoms of species j in structure i
        # total_energies: a 1D tensor of shape (num_structures,)
        #   where total_energies[i] is the total energy of structure i
        distinct_species = torch.unique(torch.cat([g.Z for g in graphs]))
        num_atoms = torch.zeros(len(graphs), len(distinct_species))
        for i, g in enumerate(graphs):
            for j, z in enumerate(distinct_species):
                num_atoms[i, j] = (g.Z == z).sum()
        total_energies = torch.tensor([g.labels["energy"] for g in graphs])

        # fit the energy offset and scale
        offsets, scales = guess_offsets_and_scales(num_atoms, total_energies)

        new_offsets = torch.zeros_like(self.energy_offset)
        new_offsets[distinct_species] = offsets
        self.energy_offset = new_offsets

        new_scales = torch.ones_like(self.energy_scale)
        new_scales[distinct_species] = scales
        self.energy_scale = new_scales

        forces = torch.cat([g.labels["forces"] for g in graphs])
        Zs = torch.cat([g.Z for g in graphs])
        _scales = self.energy_scale[Zs]
        f_std = forces / _scales.unsqueeze(-1)
        eta = torch.ones_like(self.force_eta)
        for Z in distinct_species:
            eta[Z] = (f_std[Zs == Z].flatten()).std()

        self.force_eta = eta


def guess_offsets_and_scales(
    num_atoms: torch.Tensor,
    total_energies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # offsets is easy: just do linear regression
    offsets = torch.linalg.lstsq(num_atoms, total_energies).solution

    # now to guess the scales
    mean_energies = (num_atoms @ offsets).squeeze()
    sample_variances = (total_energies - mean_energies).pow(2)

    var = torch.linalg.lstsq(num_atoms, sample_variances).solution
    var[var < 0] = 0.0001  # avoid negative variances

    return offsets, var.sqrt()
