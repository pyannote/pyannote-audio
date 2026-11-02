.. pyannote.core documentation master file, created by
   sphinx-quickstart on Thu Jan 19 13:25:34 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#############
pyannote.core
#############

`pyannote.core` is an open-source Python library providing advanced data structures for handling temporal segments with attached labels.

.. plot:: pyplots/introduction.py

It is the foundation of the `pyannote` ecosystem, made of a (growing) number of libraries ( `pyannote.parser`, `pyannote.metrics`, `pyannote.database`, `pyannote.audio`, `pyannote.video`).

It also provides a convenient way to visualize this temporal data in IPython notebooks.


Installation
============

::

$ conda create -n pyannote python=3.5 anaconda
$ source activate pyannote
$ pip install pyannote.core



User guide
==========

.. toctree::
   :maxdepth: 2

   structure
   visualization


API documentation
=================

.. toctree::
   :maxdepth: 2

   reference
   Changelog <https://github.com/pyannote/pyannote-core/blob/main/CHANGELOG.md>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
