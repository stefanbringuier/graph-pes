Distance Expansions
===================

Available Expansions
--------------------

`graph-pes` exposes the :class:`DistanceExpansion <graph_pes.models.distances.DistanceExpansion>` 
base class, which can be used to implement new distance expansions.
We also provide a few common expansions:

.. autoclass :: graph_pes.models.distances.Bessel
.. autoclass :: graph_pes.models.distances.GaussianSmearing


Implementing a new Expansion
----------------------------

.. autoclass :: graph_pes.models.distances.DistanceExpansion
    :members: