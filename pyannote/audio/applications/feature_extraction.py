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

"""
Feature extraction

Usage:
  pyannote-speech-feature [--robust --parallel] <experiment_dir> <database.task.protocol>
  pyannote-speech-feature check <experiment_dir> <database.task.protocol>
  pyannote-speech-feature -h | --help
  pyannote-speech-feature --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --robust                   When provided, skip files for which feature extraction fails.
  --parallel                 When provided, process files in parallel.
  -h --help                  Show this screen.
  --version                  Show version.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process
    (e.g. MFCCs).

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: LibrosaMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives
    ...................................................................

"""

import yaml
import os.path
import numpy as np
import functools
from docopt import docopt

from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier
from pyannote.database import get_protocol

from pyannote.core.utils.helper import get_class_by_name

from pyannote.audio.features import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.precomputed import PyannoteFeatureExtractionError

from multiprocessing import cpu_count, Pool


def init_feature_extraction(experiment_dir):

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    FeatureExtraction = get_class_by_name(
        config['feature_extraction']['name'],
        default_module_name='pyannote.audio.features')
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    return feature_extraction

def process_current_file(current_file, file_finder=None, precomputed=None,
                         feature_extraction=None, robust=False):

    try:
        current_file['audio'] = file_finder(current_file)
    except ValueError as e:
        if not robust:
            raise PyannoteFeatureExtractionError(*e.args)
        return e

    uri = get_unique_identifier(current_file)
    path = precomputed.get_path(current_file)

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

    if np.any(np.isnan(features.data)):
        msg = 'Feature extraction returned NaNs for file "{uri}".'
        return msg.format(uri=uri)

    precomputed.dump(current_file, features)

    return


def helper_extract(current_file, file_finder=None, experiment_dir=None,
                   config_yml=None, feature_extraction=None,
                   robust=False):

    if feature_extraction is None:
        feature_extraction = init_feature_extraction(experiment_dir)

    precomputed = Precomputed(root_dir=experiment_dir)
    return process_current_file(current_file, file_finder=file_finder,
                                precomputed=precomputed,
                                feature_extraction=feature_extraction,
                                robust=robust)

def extract(protocol_name, file_finder, experiment_dir,
            robust=False, parallel=False):

    protocol = get_protocol(protocol_name, progress=False)

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    FeatureExtraction = get_class_by_name(
        config['feature_extraction']['name'],
        default_module_name='pyannote.audio.features')
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    sliding_window = feature_extraction.sliding_window
    dimension = feature_extraction.dimension

    # create metadata file at root that contains
    # sliding window and dimension information

    precomputed = Precomputed(root_dir=experiment_dir,
                              sliding_window=sliding_window,
                              dimension=dimension)

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
        imap = map


    for result in imap(extract_one, protocol.files()):
        if result is None:
            continue
        print(result)

def check(protocol_name, file_finder, experiment_dir):

    protocol = get_protocol(protocol_name)
    precomputed = Precomputed(experiment_dir)

    for subset in ['development', 'test', 'train']:

        try:
            file_generator = getattr(protocol, subset)()
            first_item = next(file_generator)
        except NotImplementedError as e:
            continue

        for current_file in getattr(protocol, subset)():

            try:
                audio = file_finder(current_file)
                current_file['audio'] = audio
            except ValueError as e:
                print(e)
                continue

            duration = get_audio_duration(current_file)

            try:
                features = precomputed(current_file)
            except PyannoteFeatureExtractionError as e:
                print(e)
                continue

            if not np.isclose(duration,
                              features.getExtent().duration,
                              atol=1.):
                uri = get_unique_identifier(current_file)
                print('Duration mismatch for "{uri}"'.format(uri=uri))

            if np.any(np.isnan(features.data)):
                uri = get_unique_identifier(current_file)
                print('NaN for "{uri}"'.format(uri=uri))


def main():

    arguments = docopt(__doc__, version='Feature extraction')

    file_finder = FileFinder()

    protocol_name = arguments['<database.task.protocol>']
    experiment_dir = arguments['<experiment_dir>']

    if arguments['check']:
        check(protocol_name, file_finder, experiment_dir)
    else:
        robust = arguments['--robust']
        parallel = arguments['--parallel']
        extract(protocol_name, file_finder, experiment_dir,
                robust=robust, parallel=parallel)
