.. _models:

######
Models
######

All models implemented in ``graph_pes`` are subclasses of
:class:`graph_pes.core.GraphPESModel`. 


.. autoclass :: graph_pes.core.GraphPESModel
   :members: predict_local_energies, __add__


Pair Potentials
===============

Pair potential models can be recast as local-energy models acting on graphs.

.. autoclass :: graph_pes.models.pairwise.PairPotential
   :members: interaction

.. autoclass :: graph_pes.models.pairwise.LennardJones
