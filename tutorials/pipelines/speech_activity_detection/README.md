> The MIT License (MIT)
>
> Copyright (c) 2019 CNRS
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
> AUTHOR
> Hervé Bredin - http://herve.niderb.fr

# Speech activity detection pipeline with `pyannote.audio`

Training a model for speech activity detection is not enough to get actual speech activity detection results. One has to also tune detection thresholds (and other optional pipeline hyper-parameters).

In this tutorial, you will learn how to optimize a speech activity detection pipeline using `pyannote-pipeline` command line tool.

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speech activity detection, please cite the following paper:

```bibtex
@inproceedings{Yin2018,
  Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  Title = {{Neural Speech Turn Segmentation and Affinity Propagation for Speaker Diarization}},
  Booktitle = {{19th Annual Conference of the International Speech Communication Association, Interspeech 2018}},
  Year = {2018},
  Month = {September},
  Address = {Hyderabad, India},
}
```

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-pipeline` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/pipelines/speech_activity_detection/config.yml
```
```yaml
pipeline:
   name: pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection
   params:
      scores: /path/to/precomputed/sad
```

This configuration file assumes that you have already been through the speech actitity detection (model) tutorial and applied it into `/path/to/precomputed/sad`.
 
## Training
([↑up to table of contents](#table-of-contents))

The following command will run hyper-parameter optimization on the development subset of the AMI database:

```bash
$ export EXPERIMENT_DIR=tutorials/pipelines/speech_activity_detection
$ pyannote-pipeline train --forever ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can run this command on several machines in parallel to speed up the hyper-parameter search.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.development
$ cat ${TRAIN_DIR}/params.yml
```
```yaml
min_duration_off: 0.6857137236312955
min_duration_on: 0.3225952679776678
offset: 0.9436397097473367
onset: 0.704966228813754
pad_offset: 0.08311274833799132
pad_onset: 0.06505433882746965
```

See `pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection` docstring for details about these hyper-parameters.

## Application
([↑up to table of contents](#table-of-contents))

The optimized pipeline can then be applied on all files of the AMI database:

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.development
$ pyannote-pipeline apply ${TRAIN_DIR}/params.yml AMI.SpeakerDiarization.MixHeadset /path/to/pipeline/output
```

## More options

For more options, see:

```bash
$ pyannote-pipeline --help
```

That's all folks!
