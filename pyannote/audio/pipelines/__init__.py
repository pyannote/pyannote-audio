# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .overlapped_speech_detection import OverlappedSpeechDetection
from .resegmentation import Resegmentation
from .segmentation import Segmentation
from .speaker_diarization import SpeakerDiarization
from .voice_activity_detection import VoiceActivityDetection

# from .speaker_change_detection import SpeakerChangeDetection  #  not optimizable
# from .speech_turn_assignment import SpeechTurnClosestAssignment  # not optimizable
# from .speech_turn_clustering import SpeechTurnClustering  #  not optimizable
# from .speech_turn_segmentation import SpeechTurnSegmentation  # not optimizable

__all__ = [
    "VoiceActivityDetection",
    "Segmentation",
    "OverlappedSpeechDetection",
    "SpeakerDiarization",
    "Resegmentation",
]
