#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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


from tqdm import tqdm
from .base import Application
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.features import Precomputed
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.core.utils.helper import get_class_by_name
from functools import partial
import multiprocessing as mp


class BaseLabeling(Application):

    def __init__(self, experiment_dir, db_yml=None, training=False):

        super(BaseLabeling, self).__init__(
            experiment_dir, db_yml=db_yml, training=training)

        # task
        Task = get_class_by_name(
            self.config_['task']['name'],
            default_module_name='pyannote.audio.labeling.tasks')
        self.task_ = Task(
            **self.config_['task'].get('params', {}))

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.labeling.models')
        self.get_model_ = partial(
            Architecture,
            **self.config_['architecture'].get('params', {}))

    def validate_init(self, protocol_name, subset='development'):

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        self.pool_ = mp.Pool(mp.cpu_count())

        if isinstance(self.feature_extraction_, Precomputed):
            return list(files)

        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'):
            current_file['features'] = self.feature_extraction_(current_file)
            validation_data.append(current_file)

        return validation_data

    def apply(self, protocol_name, output_dir, step=None, subset=None):

        model = self.model_.to(self.device)
        model.eval()

        duration = self.task_.duration
        if step is None:
            step = 0.25 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

        sliding_window = sequence_labeling.sliding_window

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            labels=model.classes)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        if subset is None:
            files = FileFinder.protocol_file_iter(protocol,
                                                  extra_keys=['audio'])
        else:
            files = getattr(protocol, subset)()

        for current_file in files:
            fX = sequence_labeling(current_file)
            precomputed.dump(current_file, fX)
