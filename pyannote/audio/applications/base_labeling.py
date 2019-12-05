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

from typing import Optional
from pathlib import Path
from tqdm import tqdm
from .base import Application
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.audio.features import Precomputed
from pyannote.audio.features import RawAudio
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.core.utils.helper import get_class_by_name
from functools import partial
import multiprocessing as mp


class BaseLabeling(Application):

    @property
    def config_main_section(self):
        return 'task'

    @property
    def config_default_module(self):
        return 'pyannote.audio.labeling.tasks'

    def validate_init(self, protocol_name, subset='development'):
        """Initialize validation data

        Parameters
        ----------
        protocol_name : `str`
        subset : {'train', 'development', 'test'}
            Defaults to 'development'.

        Returns
        -------
        validation_data : object
            Validation data.

        """

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        n_jobs = getattr(self, 'n_jobs', 1)
        self.pool_ = mp.Pool(n_jobs)

        if isinstance(self.feature_extraction_, (Precomputed, RawAudio)):
            return list(files)

        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'):
            current_file['features'] = self.feature_extraction_(current_file)
            validation_data.append(current_file)

        return validation_data

    def apply(self, protocol_name: str,
                    step: Optional[float] = None,
                    subset: Optional[str] = "test",
                    return_intermediate: Optional[int] = None):
        """Apply pre-trained model



        Parameters
        ----------
        protocol_name : `str`
        step : `float`, optional
            Time step. Defaults to 25% of sequence duration.
        subset : {'train', 'development', 'test'}
            Defaults to 'test'
        return_intermediate : `int`, optional
            Index of intermediate layer. Returns intermediate hidden state.
            Defaults to returning the final output.
        """

        model = self.model_.to(self.device)
        model.eval()

        duration = self.task_.duration
        if step is None:
            step = 0.25 * duration

        output_dir = Path(self.APPLY_DIR.format(
            validate_dir=self.validate_dir_,
            epoch=self.epoch_))

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device, return_intermediate=return_intermediate)

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sequence_labeling.sliding_window,
            labels=model.classes)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for current_file in getattr(protocol, subset)():
            fX = sequence_labeling(current_file)
            precomputed.dump(current_file, fX)

        # do not proceed with the full pipeline
        # when there is no such thing for current task
        if not hasattr(self, 'Pipeline'):
            return

        # instantiate pipeline
        pipeline = self.Pipeline(scores=output_dir)
        pipeline.instantiate(self.pipeline_params_)

        # load pipeline metric (when available)
        try:
            metric = pipeline.get_metric()
        except NotImplementedError as e:
            metric = None

        # apply pipeline and dump output to RTTM files
        output_rttm = output_dir / f'{protocol_name}.{subset}.rttm'
        with open(output_rttm, 'w') as fp:
            for current_file in getattr(protocol, subset)():
                hypothesis = pipeline(current_file)
                pipeline.write_rttm(fp, hypothesis)

                # compute evaluation metric (when possible)
                if 'annotation' not in current_file:
                    metric = None

                # compute evaluation metric (when available)
                if metric is None:
                    continue

                reference = current_file['annotation']
                uem = get_annotated(current_file)
                _ = metric(reference, hypothesis, uem=uem)

        # print pipeline metric (when available)
        if metric is None:
            return

        output_eval = output_dir / f'{protocol_name}.{subset}.eval'
        with open(output_eval, 'w') as fp:
            fp.write(str(metric))
