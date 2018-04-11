#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import versioneer

from setuptools import setup, find_packages

setup(

    # package
    namespace_packages=['pyannote'],
    packages=find_packages(),
    install_requires=[
        'pyannote.core >= 1.3.1',
        'pyannote.metrics >= 1.7',
        'pyannote.generators >= 0.17.1',
        'pyannote.database >= 1.3',
        'pyannote.parser >= 0.7',
        'pyannote.algorithms >= 0.7.3',
        'pandas >= 0.18.0',
        'audioread >= 2.1.5',
        'librosa >= 0.5.1',
        'sphfile == 1.0.0',
        'scikit-optimize >= 0.2',
        'pyYAML >= 3.12',
        'h5py >= 2.6.0',
        'cachetools >= 2.0.0',
        'tqdm >= 4.11.2',
        'sortedcontainers >= 1.5.9',
        'sortedcollections >= 0.5.3',
        'tensorboardX >= 1.1',
        'dlib == 19.10.0',
    ],
    entry_points = {
        'console_scripts': [
            'pyannote-speech-feature=pyannote.audio.applications.feature_extraction:main',
            'pyannote-speech-detection=pyannote.audio.applications.speech_detection:main',
            'pyannote-change-detection=pyannote.audio.applications.change_detection:main',
            'pyannote-overlap-detection=pyannote.audio.applications.overlap_detection:main',
            'pyannote-speaker-embedding=pyannote.audio.applications.speaker_embedding:main',
            'pyannote-bic-clustering=pyannote.audio.applications.bic_clustering:main']
    },
    # versioneer
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    # PyPI
    name='pyannote.audio',
    description=('Audio processing'),
    author='Hervé Bredin',
    author_email='bredin@limsi.fr',
    url='http://herve.niderb.fr/',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"
    ],
)
