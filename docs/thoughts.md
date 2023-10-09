# `GraphPES`

`GraphPES` is a framework for models that act on graph representations of atomic structures to predict total energies (and therefore forces).

This is not a trivial task. `GraphPES` attempts to remain as general as possible, while still providing a simple interface for training and evaluating models, i.e. making the process just as complicated as it needs to be, and no more. We provide sensible defaults, and allow for easy customization.

---

## Why graphs?

Techniques exist (DFT, CCSD(T) etc.) to "exactly" calculate the potential energy surface (PES) of a system.
Given that the atomic coordinates $\mathbf{R}$ and numbers $Z$ are what uniquely define a system, these methods can be considered as functions that act on pointclouds to generate total energies:
$$E = E(\{\mathbf{R}_i, Z_i\})$$
Unfortunately, these techniques are computationally expensive, scaling polynomially with the number of atoms, $N$, and having large prefactors, _i.e._ they are slow/expensive for small systems, and get catastrophically more so as $N$ increases.

To model the PES of a system in a way that scales linearly with system size (the best possible scaling), a two part assumption is frequently made:

1. The total system energy can be written as a sum of atomic contributions: $$E = \sum_i^N \varepsilon_i$$
2. These atomic contributions, $\varepsilon_i$ are local in nature, _i.e._, only depend on (small) local neighbourhoods, $\mathcal{N}_i$, and hence are constant wrt. system size to compute, $O(1)$ $$\varepsilon_i = \varepsilon_i(\{\mathbf{R}_j, Z_j\}_{j \in \mathcal{N}_i})$$

A natural way to incorporate such locality within a global structure is to transition from a pointcloud to a graph representation, $\mathcal{G}$, where:

-   each atom is represented by a node
    -   a single node feature, $Z_i$, holds its atomic number
-   a (directed) edge from node $i$ to node $j$ exists $\text{iff} \,\, j \in \mathcal{N}_i$
    -   a single edge feature, $\vec{r}_{ij}$, holds the vector from atom $i$ to atom $j$

Relative position vectors are stored on edges rather than nodes so as to (a) make the graph representation (and hence any function acting on it) translation invariant, and (b) allow for the use of periodic boundary conditions, where $\vec{r}_{ij}$ is not necessarily given by $\mathbf{R}_j - \mathbf{R}_i$ and where multiple edges with different relative position vectors may exist between a given pair of atoms.

> A natural way to define local neighbourhoods is via a distance cutoff, $|\vec{r}_{ij}| < r_{\rm cutoff}$. This is not the only way, however, and alternatives exist.

---

## Training

Most modern (deep) machine learning relies on the use of **standardized** data, _i.e._, data with zero mean and unit variance.
Making predictions in, and transforming ground truth labels into, such a standardized space has a number of advantages:

-   **popular loss functions** (e.g. MSE) produce "nice" learning signals, _i.e._, losses (and therefore gradients) are not too small/large, and integrate nicely with:
-   **traditional optimizers** (e.g. Adam) work well with their standard hyperparameters (e.g. learning rate, weight decay, etc.)
-   model performance is much less dependent on **hyperparameter selection** when moving from one dataset to another, since the training signal is independent of any global shift and scaling of the data

During the training process of a `GraphPES` model, we therefore transform the ground truth labels and model predictions into a standardized space in order to calculate losses and update the model parameters. The manner in which we do this is set out below.

---

## Energies

Standardizing total energies is non-trivial yet essential: we want to ensure that the standardized ground truth energies, $E_{\text{std}}^{\text{true}}$ are as close to normally distributed as possible (as taken over the training set), _i.e._, have zero mean and unit variance. Our standardisation procedure therefore needs to account for:

-   structures of different sizes
-   structures of varying chemical composition

A natural way to do this is to again assume that the total energy can be written as a sum of atomic contributions, $E = \sum_i^N \varepsilon_{Z_i}$, and that local energy contributions per species are normally distributed (in the absence of any other information):

$$
\varepsilon_{Z} \sim \mathcal{N}(\mu_Z, \sigma_Z^2)
$$

The total energy for a structure is distributed as:

$$
E \sim \mathcal{N}\left(\sum_i \mu_{Z_i}, \sqrt{\sum_i \sigma_{Z_i}^2}\right)
$$

Once we have the per-species mean and variance of the training set energies, $\mu_Z$ and $\sigma_Z^2$, we interconvert between the standardized and regular spaces using the following transforms:

$$
\begin{align*}
T : E \rightarrow E_{\rm std} &= (E - \sum_i\mu_{Z_i}) / \sqrt{\sum_i \sigma_{Z_i}^2} \\
\implies T^{-1} : E_{\rm std} \rightarrow E &= \sqrt{\sum_i \sigma_{Z_i}^2} \cdot E_{\rm std} + \sum_i\mu_{Z_i}
\end{align*}
$$

All `GraphPES` models, $f$:

-   implement a function that predicts standardised local energies, $\varepsilon_i$, for each atom in a structure, as represented by $\mathcal{G}$
    -   `<model>.predict_local_energies(graph: AtomicGraph) -> torch.Tensor`
-   own a transform that post-processes this standardized prediction into the final total system energy: $$\hat{E} = \sum_{i \, \in \, \mathcal{G}} \lambda_{Z_i} \varepsilon_i + \varepsilon_{0,Z_i}$$ where the per-species scaling factor, $\lambda$, and offset, $\varepsilon_0$, can be fit before, and optionally optimized during, training.
    -   `<model>.post_process(local_energies: torch.Tensor) -> torch.Tensor`

Typically, we only have access to the ground truth total energy, $E^{\rm true}$, rather than any local energy contributions. To train the model in the standardized space, we therefore need to transform the ground truth labels.

To model the total system energy, all `GraphPES` models therefore generate scalar, standardized predictions $\hat{E}_{\rm std} = f_{\rm std}(\mathcal{G})$, which can then be post-processed internally to generate the final energy prediction $$\hat{E} = \lambda \hat{E}_{\rm std} + \sum_{i \, \in \, \mathcal{G}} \varepsilon_{0,Z_i}$$

where $\lambda$ is a scaling factor, $\varepsilon_{0,Z_i}$ is a per-species energy offset, and both can be fit before, and (optionally) optimized during, training.

Total energy is an extensive property, _i.e._, it scales with the number of atoms in the system.
However, we want the magnitude of an energy loss to be independent of system size so as not to artificially de/increase the training signal to dis/favour accuracy on different sized systems.

To train in the standardized space, we therefore apply loss metrics to the standardized energy predictions, as scaled by $\sqrt{N}$ according to the central limit theorem:

$$
L_E = \text{metric}(\hat{E}_{\rm std} / \sqrt{N}, E_{\rm std}^{\rm true} / \sqrt{N})
$$

where $N$ is the number of atoms in the system.

We generate the standardized ground truth labels using the pre-calculated per-atom-mean and variance of the training set energies, $\mu_{Z}$ and $\sigma^2$, to use in the fixed transforms:

$$
\begin{align*}
E_{\rm std}^{\rm true} &= (E^{\rm true} - \sum_{i\, \in \, \mathcal{G}}\mu_{Z_I}) / \sigma \\
\implies F_{\rm std}^{\rm true} &= F^{\rm true} / \sigma
\end{align*}
$$

---

## Forces

By definition, the forces on each atom are the negative gradients of the energy with respect to the atomic coordinates, _i.e._ $\mathbf{F} = -\nabla_{R} E$.

We can therefore make use of the autograd machinery to calculate these automatically, both in the standardized and regular space (without the need to directly model them):

$$
\begin{align*}
\hat{\mathbf{F}} &= -\nabla_{\hat{R}} \hat{E} \\
\hat{\mathbf{F}}_{\rm std} &= -\nabla_{R} \hat{E}_{\rm std} = - \nabla_{R} \hat{E} / \lambda =  \hat{\mathbf{F}} / \lambda
\end{align*}
$$

One final complexity here is that the magnitude of the forces derived from the standardized energy predictions, $\| \hat{\mathbf{F}}_{\rm std} \|$, are in no way guaranteed to have unit variance.

We therefore use an additional scaling factor, $\eta_F$, in all force-based loss functions:

$$
L_F = \text{metric}(\hat{\mathbf{F}}_{\rm std}/\eta_f, \, \mathbf{F}_{\rm std}^{\rm true} / \eta_f)
$$

such that the magnitudes $\| \hat{\mathbf{F}}_{\rm std} \| / \eta_f = \| \hat{\mathbf{F}} \| / (\lambda \cdot \eta_F)$ **do** have unit variance. (Again we also report on metrics applied to the processed predictions for convenience, but these are not used to update the model.)

---

## Implementation

Before training occurs, the following quantities need to be calculated:

-   $\mu_E$ and ${\sigma^2}_E$ : the mean and variance of the training set energies
    -   used to generate the standardized ground truth labels
-   $\eta_F = \sigma\left( \|\mathbf{F}_{\rm std}^{\rm true} \| \right)$ : the additional force scaling factor
    -   used to further scale force inputs to loss functions

These are bundled into a `Standardizer` object.

To calculate initial values for $\lambda$ and $\varepsilon_{0,Z_i}$, we use the following procedure:

-   $\lambda \rightarrow \sigma_E$ : set $\lambda$ to the standard deviation of the training set energies
-   $\varepsilon_{0,Z_i} \rightarrow \mu_E$ : set $\varepsilon_{0,Z_i}$ to the mean of the training set energies

The inputs to the training loop are:

-   `model` : a `GraphPES` model that can:
    -   predict the $\hat{E}_{\rm std}$ for a graph, $\mathcal{G}$, implemented as `model.predict_energy(graph: AtomicGraph) -> torch.Tensor`
    -   post-process $\hat{E}_{\rm std}$ into the final $\hat{E}$, implemented as `model.post_process(energy_std: torch.Tensor) -> torch.Tensor`
-   $E^{\rm true}, \{{\mathbf{F}_i}^{\rm true} \}$ : the ground truth labels

In pseudocode, the training loop looks like:

```python
# 1. generate standardized predictions
E_std, F_std = energy_and_forces(model.predict_energy, graph)

# 2. energy loss
```
