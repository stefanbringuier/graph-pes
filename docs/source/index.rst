
.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:

   Home <self>
   models
   data
   training
   analysis
   examples

################################
:code:`graph-pes` Documentation
################################


:code:`graph-pes` is a framework for building, training and analysing potential energy surfaces (PESs) that act on
graph representations of atomic structures.

Getting Started
===============

.. code-block:: bash

    pip install graph-pes

:code:`graph-pes` aims to provide sensible defaults to allow for getting started quickly:

.. literalinclude:: quickstart.py
    :language: python

For a more detailed introduction, see an example notebook `here <notebooks/example.html>`_. 
For under-the-hood details see :func:`train_model() <graph_pes.training.train_model>`.

