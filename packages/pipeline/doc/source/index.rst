
#################
pyannote.pipeline
#################

`pyannote.pipeline` is an open-source Python library for pipeline optimization

See the `changelog <https://github.com/pyannote/pyannote-pipeline/blob/main/CHANGELOG.md>`_
for release notes and compatibility changes.


Installation
============

::

$ conda create -n pyannote python=3.6 anaconda
$ source activate pyannote
$ pip install --process-dependency-links pyannote.pipeline

If on MacOS, an `ghalton` fails to install, you might need to do something like

::

$ export CFLAGS="-Wno-deprecated-declarations -std=c++11 -stdlib=libc++"
$ pip install ghalton
