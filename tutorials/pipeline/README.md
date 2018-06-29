> The MIT License (MIT)
>
> Copyright (c) 2018 CNRS
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

# Speaker diarization pipeline with `pyannote.audio`

In this tutorial, you will learn how to optimize a speaker diarization pipeline using `pyannote-pipeline` command line tool.

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speaker diarization, please cite the following paper:

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
$ cat tutorials/pipeline/config.yml
```
```yaml
pipeline:
   name: Yin2018
   params:
      sad: tutorials/pipeline/sad
      scd: tutorials/pipeline/scd
      emb: tutorials/pipeline/emb
      metric: angular

sampler:
   name: CMAES
```

This configuration file assumes that you have already been through the other tutorials and applied
  - speech activity detection (into `tutorials/pipeline/sad`)
  - speaker change detection (into `tutorials/pipeline/scd`)
  - speaker embedding (into `tutorials/pipeline/emb`)

## Training
([↑up to table of contents](#table-of-contents))

The following command will run hyper-parameter optimization on the development subset of the AMI database:

```bash
$ export EXPERIMENT_DIR=tutorials/pipeline
$ pyannote-pipeline train --forever ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can follow along the training process using [tensorboard](https://github.com/tensorflow/tensorboard).
```bash
$ tensorboard --logdir=${EXPERIMENT_DIR}
```

One can run this command on several machines in parallel to speed up the hyper-parameter search.

## Application
([↑up to table of contents](#table-of-contents))

The optimized pipeline can then be applied on all files of the AMI database:

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.train
$ pyannote-pipeline apply ${TRAIN_DIR}/params.yml AMI.SpeakerDiarization.MixHeadset /path/to/pipeline/output
```

## More options

For more options, see:

```bash
$ pyannote-pipeline --help
```

That's all folks!
