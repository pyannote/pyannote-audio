#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

"""
Feature extraction

Usage:
  pyannote-speech-feature [--robust --parallel --database=<db.yml>] <experiment_dir> <database.task.protocol>
  pyannote-speech-feature -h | --help
  pyannote-speech-feature --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --robust                   When provided, skip files for which feature extraction fails.
  --parallel                 When provided, process files in parallel.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process
    (e.g. MFCCs).

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives
    ...................................................................

"""

import yaml
import h5py
import os.path
import numpy as np
import functools
import itertools
from docopt import docopt

import pyannote.core
import pyannote.database
from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier
from pyannote.database import get_protocol

from pyannote.audio.util import mkdir_p
from pyannote.audio.features.utils import Precomputed
from pyannote.audio.features.utils import PyannoteFeatureExtractionError

from multiprocessing import cpu_count, Pool


def init_feature_extraction(experiment_dir):

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    return feature_extraction

def process_current_file(current_file, file_finder=None, experiment_dir=None,
                         feature_extraction=None):

    try:
        current_file['audio'] = file_finder(current_file)
    except ValueError as e:
        if not robust:
            raise PyannoteFeatureExtractionError(*e.args)
        return e

    uri = get_unique_identifier(current_file)
    path = Precomputed.get_path(experiment_dir, current_file)

    if os.path.exists(path):
        return

    try:
        features = feature_extraction(current_file)
    except PyannoteFeatureExtractionError as e:
        msg = 'Feature extraction failed for file "{uri}".'
        return msg.format(uri=uri)

    if features is None:
        msg = 'Feature extraction returned None for file "{uri}".'
        return msg.format(uri=uri)

    data = features.data

    if np.any(np.isnan(data)):
        msg = 'Feature extraction returned NaNs for file "{uri}".'
        return msg.format(uri=uri)

    # create parent directory
    mkdir_p(os.path.dirname(path))

    sliding_window = feature_extraction.sliding_window()
    dimension = feature_extraction.dimension()

    f = h5py.File(path)

    f.attrs['start'] = sliding_window.start
    f.attrs['duration'] = sliding_window.duration
    f.attrs['step'] = sliding_window.step
    f.attrs['dimension'] = dimension
    f.create_dataset('features', data=data)
    f.close()

    return


def helper_extract(current_file, file_finder=None, experiment_dir=None,
                   config_yml=None, feature_extraction=None, robust=False):

    if feature_extraction is None:
        feature_extraction = init_feature_extraction(config_yml, experiment_dir)

    return process_current_file(current_file, file_finder=file_finder,
                                experiment_dir=experiment_dir,
                                feature_extraction=feature_extraction)

def extract(protocol_name, file_finder, experiment_dir,
            robust=False, parallel=False):

    protocol = get_protocol(protocol_name, progress=False)

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    sliding_window = feature_extraction.sliding_window()
    dimension = feature_extraction.dimension()

    # create metadata file at root that contains
    # sliding window and dimension information
    path = Precomputed.get_config_path(experiment_dir)
    f = h5py.File(path)
    f.attrs['start'] = sliding_window.start
    f.attrs['duration'] = sliding_window.duration
    f.attrs['step'] = sliding_window.step
    f.attrs['dimension'] = dimension
    f.close()

    if parallel:

        extract_one = functools.partial(helper_extract,
                                        file_finder=file_finder,
                                        experiment_dir=experiment_dir,
                                        config_yml=config_yml,
                                        robust=robust)

        n_jobs = cpu_count()
        pool = Pool(n_jobs)
        imap = pool.imap

    else:

        feature_extraction = init_feature_extraction(experiment_dir)
        extract_one = functools.partial(helper_extract,
                                        file_finder=file_finder,
                                        experiment_dir=experiment_dir,
                                        feature_extraction=feature_extraction,
                                        robust=robust)

        imap = itertools.imap

    for subset in ['development', 'test', 'train']:

        try:
            protocol.progress = False
            file_generator = getattr(protocol, subset)()
            first_item = next(file_generator)
        except NotImplementedError as e:
            continue

        protocol.progress = True
        file_generator = getattr(protocol, subset)()

        for result in imap(extract_one, file_generator):
            if result is None:
                continue
            print(result)


def main():

    arguments = docopt(__doc__, version='Feature extraction')

    db_yml = os.path.expanduser(arguments['--database'])
    file_finder = FileFinder(db_yml)

    protocol_name = arguments['<database.task.protocol>']
    experiment_dir = arguments['<experiment_dir>']

    robust = arguments['--robust']
    parallel = arguments['--parallel']
    extract(protocol_name, file_finder, experiment_dir,
            robust=robust, parallel=parallel)
