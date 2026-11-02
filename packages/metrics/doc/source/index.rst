.. pyannote.metrics documentation master file, created by
   sphinx-quickstart on Thu Jan 19 11:54:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################
pyannote.metrics
################

"A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems"
------------------------------------------------------------------------------------------------------

`pyannote.metrics` is an open-source Python library aimed at researchers working in the wide area of speaker diarization. It provides a command line interface (CLI) to improve reproducibility and comparison of speaker diarization research results. Through its application programming interface (API), a large set of evaluation metrics is available for diagnostic purposes of all modules of typical speaker diarization pipelines (speech activity detection, speaker change detection, clustering, and identification). Finally, thanks to `pyannote.core` visualization capabilities, it can also be used for detailed error analysis purposes.


Installation
============

::

$ pip install pyannote.metrics

Citation
========

If you use `pyannote.metrics` in your research, please use the following citation:

::

  @inproceedings{pyannote.metrics,
    author = {Herv\'e Bredin},
    title = {{pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems}},
    booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
    year = {2017},
    month = {August},
    address = {Stockholm, Sweden},
    url = {http://pyannote.github.io/pyannote-metrics},
  }


User guide
==========

.. toctree::
   :maxdepth: 3

   api
   cli

API documentation
=================

.. toctree::
   :maxdepth: 3

   reference
   Changelog <https://github.com/pyannote/pyannote-metrics/blob/main/CHANGELOG.md>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
