#####
Data
#####

Atomic Graphs
=============

An :class:`AtomicGraph<graph_pes.data.AtomicGraph>` is a representation of an atomic structure. Each node 
corresponds to an atom, and each edge links two atoms that are "bonded". 

The simplest way to create such a graph is to use 
:func:`convert_to_atomic_graph<graph_pes.data.convert_to_atomic_graph>`, 
which converts an ASE atoms object into an :class:`AtomicGraph<graph_pes.data.AtomicGraph>`.


(mention pbcs + complete info required + maybe a diagram?)

.. autoclass :: graph_pes.data.AtomicGraph
    :members:

.. autofunction :: graph_pes.data.convert_to_atomic_graph


Batching
========

.. autoclass :: graph_pes.data.AtomicGraphBatch
    :members:

.. autoclass :: graph_pes.data.AtomicDataLoader
    :members:

.. autofunction :: graph_pes.data.sum_per_structure