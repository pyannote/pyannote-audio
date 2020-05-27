#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2019 CNRS

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

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pyannote.audio",
    namespace_packages=["pyannote"],
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pyannote-audio=pyannote.audio.applications.pyannote_audio:main",
            "pyannote-speech-feature=pyannote.audio.applications.feature_extraction:main",
        ],
        "prodigy_recipes": [
            "pyannote.sad.manual = pyannote.audio.interactive.recipes.sad:sad_manual",
            "pyannote.dia.binary = pyannote.audio.interactive.recipes.dia:dia_binary",
            "pyannote.dia.manual = pyannote.audio.interactive.recipes.dia:dia_manual",
        ],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Neural building blocks for speaker diarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hervé Bredin",
    author_email="bredin@limsi.fr",
    url="https://github.com/pyannote/pyannote-audio",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
