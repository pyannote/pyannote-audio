# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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


from .speaker_diarization import SpeakerDiarizationProtocol


class SpeakerSpottingProtocol(SpeakerDiarizationProtocol):
    """Speaker spotting protocol

    Parameters
    ----------
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'audio': '/path/to/{uri}.wav'})
    """

    def trn_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "trn_iter".'
        )

    def trn_enrol_iter(self):
        pass

    def trn_try_iter(self):
        pass

    def dev_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "dev_iter".'
        )

    def dev_enrol_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "dev_enrol_iter".'
        )

    def dev_try_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "dev_try_iter".'
        )

    def tst_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "tst_iter".'
        )

    def tst_enrol_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "tst_enrol_iter".'
        )

    def tst_try_iter(self):
        raise NotImplementedError(
            'Custom speaker spotting protocol should implement "tst_try_iter".'
        )

    def train_enrolment(self):
        """Iterate over the enrolments of the train set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * model_id: str
          unique model identifier (the same speaker might have different models)
        * enrol_with: pyannote.core.Timeline
          parts of the audio file to use for enrolment

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> models = {}
        >>> for enrolment in protocol.train_enrolment():
        ...     # obtain path to audio file
        ...     audio = enrolment['audio']
        ...     # obtain parts of the audio file to use for enrolment
        ...     enrol_with = enrolment['enrol_with']
        ...     # this is where enrolment actually happens
        ...     model = do_something(audio, enrol_with)
        ...     # store models for later use
        ...     model_id = enrolment['model_id']
        ...     models[model_id] = model

        """

        generator = self.trn_enrol_iter()

        for current_enrolment in generator:
            yield self.preprocess(current_enrolment)

    def train_trial(self):
        """Iterate over the trials of the train set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * try_with: pyannote.core.Segment, optional
          parts of the audio file where to look for the target speaker.
          default is to use the whole audio file
        * model_id: str
          unique identifier of the target
        * reference: pyannote.core.Timeline
          parts of the audio file where the target actually speaks.
          it might be empty in case of impostor trials.
          in case of genuine trials, it should be contained in `try_with`

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> for trial in protocol.train_trial():
        ...     # obtain path to audio file
        ...     audio = trial['audio']
        ...     # obtain parts of the audio file to use for trial
        ...     try_with = trial['try_with']
        ...     # this is where the trial actually happens
        ...     model_id = trial['model_id']
        ...     score = do_something(audio, try_with, model_id)
        ...     # optionally perform evaluation
        ...     reference = trial['reference']
        ...     metric(reference, score)

        """

        generator = self.trn_try_iter()

        for current_trial in generator:
            yield self.preprocess(current_trial)

    def development_enrolment(self):
        """Iterate over the enrolments of the development set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * model_id: str
          unique model identifier (the same speaker might have different models)
        * enrol_with: pyannote.core.Timeline
          parts of the audio file to use for enrolment

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> models = {}
        >>> for enrolment in protocol.development_enrolment():
        ...     # obtain path to audio file
        ...     audio = enrolment['audio']
        ...     # obtain parts of the audio file to use for enrolment
        ...     enrol_with = enrolment['enrol_with']
        ...     # this is where enrolment actually happens
        ...     model = do_something(audio, enrol_with)
        ...     # store models for later use
        ...     model_id = enrolment['model_id']
        ...     models[model_id] = model

        """

        generator = self.dev_enrol_iter()

        for current_enrolment in generator:
            yield self.preprocess(current_enrolment)

    def development_trial(self):
        """Iterate over the trials of the development set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * try_with: pyannote.core.Segment, optional
          parts of the audio file where to look for the target speaker.
          default is to use the whole audio file
        * model_id: str
          unique identifier of the target
        * reference: pyannote.core.Timeline
          parts of the audio file where the target actually speaks.
          it might be empty in case of impostor trials.
          in case of genuine trials, it should be contained in `try_with`

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> for trial in protocol.development_trial():
        ...     # obtain path to audio file
        ...     audio = trial['audio']
        ...     # obtain parts of the audio file to use for trial
        ...     try_with = trial['try_with']
        ...     # this is where the trial actually happens
        ...     model_id = trial['model_id']
        ...     score = do_something(audio, try_with, model_id)
        ...     # optionally perform evaluation
        ...     reference = trial['reference']
        ...     metric(reference, score)

        """

        generator = self.dev_try_iter()

        for current_trial in generator:
            yield self.preprocess(current_trial)

    def test_enrolment(self):
        """Iterate over the enrolments of the test set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * model_id: str
          unique model identifier (the same speaker might have different models)
        * enrol_with: pyannote.core.Timeline
          parts of the audio file to use for enrolment

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> models = {}
        >>> for enrolment in protocol.test_enrolment():
        ...     # obtain path to audio file
        ...     audio = enrolment['audio']
        ...     # obtain parts of the audio file to use for enrolment
        ...     enrol_with = enrolment['enrol_with']
        ...     # this is where enrolment actually happens
        ...     model = do_something(audio, enrol_with)
        ...     # store models for later use
        ...     model_id = enrolment['model_id']
        ...     models[model_id] = model

        """

        generator = self.tst_enrol_iter()

        for current_enrolment in generator:
            yield self.preprocess(current_enrolment)

    def test_trial(self):
        """Iterate over the trials of the test set

        Yields dictionaries with the followings keys:

        * uri: str
          unique audio file identifier
        * database: str
          unique database identifier
        * try_with: pyannote.core.Segment, optional
          parts of the audio file where to look for the target speaker.
          default is to use the whole audio file
        * model_id: str
          unique identifier of the target
        * reference: pyannote.core.Timeline
          parts of the audio file where the target actually speaks.
          it might be empty in case of impostor trials.
          in case of genuine trials, it should be contained in `try_with`

        as well as keys coming from the provided preprocessors (e.g. 'audio')

        Usage
        -----
        >>> for trial in protocol.test_trial():
        ...     # obtain path to audio file
        ...     audio = trial['audio']
        ...     # obtain parts of the audio file to use for trial
        ...     try_with = trial['try_with']
        ...     # this is where the trial actually happens
        ...     model_id = trial['model_id']
        ...     score = do_something(audio, try_with, model_id)
        ...     # optionally perform evaluation
        ...     reference = trial['reference']
        ...     metric(reference, score)

        """

        generator = self.tst_try_iter()

        for current_trial in generator:
            yield self.preprocess(current_trial)
