#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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
Pipeline

Usage:
  pyannote-pipeline train [options] [(--forever | --trials=<trials>)] <experiment_dir> <database.task.protocol>
  pyannote-pipeline apply [options] <params.yml> <database.task.protocol> <output_dir>
  pyannote-pipeline -h | --help
  pyannote-pipeline --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset [default: development].

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  --trials=<trials>          Number of trials. [default: 1]
  --forever                  Try forever.

"apply" mode:
  <params.yml>               Path to hyper-parameters.
  <output_dir>               Directory where to store pipeline output.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml that describes the pipeline.

    ................... <experiment_dir>/config.yml ...................
    pipeline:
       name: SpeechActivityDetection
       params:
          precomputed: /path/to/sad
    ...................................................................

"train" mode:
    Tune the pipeline hyper-parameters
        <experiment_dir>/<database.task.protocol>.<subset>.yml

"apply" mode
    Finally, one can apply the pipeline using "apply" mode.

"""

import os
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os.path import dirname
from docopt import docopt
from filelock import FileLock
from .base import Application
from pyannote.core import Timeline
from tensorboardX import SummaryWriter
from pyannote.audio.util import mkdir_p
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier


class Pipeline(Application):

    @classmethod
    def from_params_yml(cls, params_yml, db_yml=None):
        train_dir = dirname(params_yml)
        app = cls.from_train_dir(train_dir, db_yml=db_yml)
        app.params_yml_ = params_yml
        with open(params_yml, mode='r') as fp:
            params = yaml.load(fp)
        app.pipeline_.with_params(**params)
        return app

    def __init__(self, experiment_dir, db_yml=None):

        super(Pipeline, self).__init__(
            experiment_dir, db_yml=db_yml)

        # pipeline
        pipeline_name = self.config_['pipeline']['name']
        pipelines = __import__('pyannote.audio.pipeline',
                               fromlist=[pipeline_name])
        self.pipeline_ = getattr(pipelines, pipeline_name)(
            **self.config_['pipeline'].get('params', {}))

    def dump(self, best, params_yml, params_yml_lock):
        content = yaml.dump(best['params'], default_flow_style=False)
        with FileLock(params_yml_lock):
            with open(params_yml, mode='w') as fp:
                fp.write(content)
        return content

    def train(self, protocol_name, subset='development', n_calls=1):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(train_dir)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        tune_db = f'{train_dir}/tune.db'
        params_yml = f'{train_dir}/params.yml'
        params_yml_lock = f'{train_dir}/params.yml.lock'

        pid = os.getpid()
        writer = SummaryWriter(log_dir=f"{train_dir}/{pid}")

        progress_bar = tqdm(unit='trial')
        progress_bar.set_description('Trial #1 : ...')
        progress_bar.update(0)

        iterations = self.pipeline_.tune_iter(tune_db, protocol, subset=subset)

        for s, status in enumerate(iterations):

            if s+1 == n_calls:
                break

            loss = status['latest']['loss']
            writer.add_scalar(f'train/{protocol_name}.{subset}/loss/latest',
                              loss, global_step=s + 1)
            if 'new_best' in status:
                _ = self.dump(status['new_best'], params_yml, params_yml_lock)
                n_trials = status['new_best']['n_trials']
                best_loss = status['new_best']['loss']
                writer.add_scalar(f'train/{protocol_name}.{subset}/loss/best',
                                  best_loss, global_step=n_trials)

            # progress bar
            desc = f"Trial #{s+1}"
            loss = status['latest']['loss']
            if abs(loss) < 1:
                desc += f" = {100 * loss:.3f}%"
                desc += f" : Best = {100 * best_loss:.3f}% after {n_trials} trials"
            else:
                desc += f" = {loss:.3f}"
                desc += f" : Best = {best_loss:.3f} after {n_trials} trials"

            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

        best = self.pipeline_.best(tune_db)
        content = self.dump(best, params_yml, params_yml_lock)

        sep = "=" * max(len(params_yml),
                        max(len(l) for l in content.split('\n')))
        print(f"\n{sep}\n{params_yml}\n{sep}\n{content}{sep}")
        print(f"Loss = {best['loss']:g} | {best['n_trials']} trials")
        print(f"{sep}")

    def apply(self, protocol_name, output_dir):

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        mkdir_p(output_dir)
        path = Path(output_dir) / f'{protocol_name}.txt'

        with open(path, mode='w') as fp:

            for current_file in FileFinder.protocol_file_iter(
                protocol, extra_keys=['audio']):

                uri = get_unique_identifier(current_file)
                hypothesis = self.pipeline_.apply(current_file)

                if isinstance(hypothesis, Timeline):
                    for s in hypothesis:
                        fp.write(f'{uri} {s.start:.3f} {s.end:.3f}\n')
                    continue

                for s, t, l in hypothesis.itertracks(yield_label=True):
                    fp.write(f'{uri} {s.start:.3f} {s.end:.3f} {t:g} {l:g}\n')

def main():

    arguments = docopt(
        __doc__, version='Pipeline hyper-parameter optimization')

    db_yml = Path(arguments['--database'])
    db_yml = db_yml.expanduser().resolve(strict=True)

    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        if arguments['--forever']:
            trials = -1
        else:
            trials = int(arguments['--trials'])

        if subset is None:
            subset = 'development'

        application = Pipeline(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset, n_calls=trials)

    if arguments['apply']:

        params_yml = Path(arguments['<params.yml>'])
        params_yml = params_yml.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        application = Pipeline.from_params_yml(
            params_yml, db_yml=db_yml)
        application.apply(protocol_name, output_dir)
