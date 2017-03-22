from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.core import SlidingWindowFeature
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from scipy.stats import zscore
import numpy as np


class ChangeDetectionBatchGenerator(PeriodicFeaturesMixin,
                                            FileBasedBatchGenerator):

    def __init__(self, feature_extractor,
                 balance=0.01, duration=3.2, step=0.8, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.balance = balance

        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source='annotated')
        super(ChangeDetectionBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):

        shape = self.shape
        dimension = 2

        return [
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': (shape[0], dimension)}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""

        # extract features for the whole file
        # (if it has not been done already)
        current_file = self.periodic_preprocess(
            current_file, identifier=identifier)

        # if labels have already been extracted, do nothing
        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        # get features as pyannote.core.SlidingWindowFeature instance
        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()

        y = np.zeros((n_samples + 4, 1), dtype=np.int8)-1
        # [-1] ==> unknown / [0] ==> not change part / [1] ==> change part

        annotated = current_file.get('annotated', X.getExtent())
        annotation = current_file['annotation']


        segments = []
        for segment, _, _ in annotation.itertracks(label=True):
            segments.append(Segment(segment.start - self.balance, segment.start + self.balance))
            segments.append(Segment(segment.end - self.balance, segment.end + self.balance))
        change_part = Timeline(segments).coverage().crop(annotated, mode='intersection')
        #coverage = annotation.get_timeline().coverage()

        # iterate over non-change regions
        for non_changes in change_part.gaps(annotated):
            indices = sw.crop(non_changes, mode='loose')
            y[indices,0] = 0

        # iterate over change regions
        for changes in change_part:
            indices = sw.crop(changes, mode='loose')
            y[indices,0] = 1

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.periodic_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)

        return [X, y]