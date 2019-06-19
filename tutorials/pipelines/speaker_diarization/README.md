> The MIT License (MIT)
>
> Copyright (c) 2018-2019 CNRS
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
$ cat tutorials/pipelines/speaker_diarization/config.yml
```
```yaml
pipeline:
   name: pyannote.audio.pipeline.speaker_diarization.SpeakerDiarization
   params:
      sad_scores: /path/to/precomputed/sad
      scd_scores: /path/to/precomputed/scd
      embedding: /path/to/precomputed/emb
      metric: cosine
      method: affinity_propagation
      evaluation_only: True

# one can freeze some of the hyper-parameters
# for instance, in this example, we are using
# hyper-parameters obtained in the speech 
# actitivy detection pipeline tutorial
freeze:
   speech_turn_segmentation:
      speech_activity_detection:
         min_duration_off: 0.6857137236312955
         min_duration_on: 0.3225952679776678
         offset: 0.9436397097473367
         onset: 0.704966228813754
         pad_offset: 0.08311274833799132
         pad_onset: 0.06505433882746965
```

This configuration file assumes that you have already been through the other tutorials and applied
  - speech activity detection (into `/path/to/precomputed/sad`)
  - speaker change detection (into `/path/to/precomputed/scd`)
  - speaker embedding (into `/path/to/precomputed/emb`)

## Training
([↑up to table of contents](#table-of-contents))

The following command will run hyper-parameter optimization on the development subset of the AMI database:

```bash
$ export EXPERIMENT_DIR=tutorials/pipelines/speaker_diarization
$ pyannote-pipeline train --forever ${EXPERIMENT_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `TRAIN_DIR` (defined below).
One can run this command on several machines in parallel to speed up the hyper-parameter search.

In particular `params.yml` contains the optimal set of hyper-parameters for this pipeline.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/AMI.SpeakerDiarization.MixHeadset.development
$ cat ${TRAIN_DIR}/params.yml
```
```yaml
min_duration: 3.9939244408136276
speech_turn_assignment:
  closest_assignment:
    threshold: 1.3786317920460274
speech_turn_clustering:
  clustering:
    damping: 0.9231549395773737
    preference: -1.571915891387201
speech_turn_segmentation:
  speaker_change_detection:
    alpha: 0.08296024595899683
    min_duration: 0.40446309879155706
  speech_activity_detection:
    min_duration_off: 0.6857137236312955
    min_duration_on: 0.3225952679776678
    offset: 0.9436397097473367
    onset: 0.704966228813754
    pad_offset: 0.08311274833799132
    pad_onset: 0.06505433882746965
```


## Application
([↑up to table of contents](#table-of-contents))

The optimized pipeline can then be applied on all files of the AMI database:

```bash
$ pyannote-pipeline apply ${TRAIN_DIR}/params.yml AMI.SpeakerDiarization.MixHeadset /path/to/pipeline/output
```
```
                    diarization error rate    total  correct correct false alarm false alarm missed detection missed detection confusion confusion
                                         %                         %                       %                                 %                   %
item
EN2002b.Mix-Headset                  44.61  2173.79  1241.47   57.11       37.32        1.72           677.86            31.18    254.46     11.71
EN2002d.Mix-Headset                  48.72  3043.00  1608.51   52.86       48.18        1.58          1087.88            35.75    346.61     11.39
ES2004a.Mix-Headset                  33.22  1051.70   723.45   68.79       21.07        2.00           265.27            25.22     62.99      5.99
ES2004b.Mix-Headset                  34.47  2403.84  1617.10   67.27       41.78        1.74           390.93            16.26    395.81     16.47
ES2004c.Mix-Headset                  26.95  2439.53  1841.84   75.50       59.69        2.45           408.44            16.74    189.25      7.76
ES2004d.Mix-Headset                  33.45  2258.51  1563.89   69.24       60.92        2.70           507.96            22.49    186.66      8.26
ES2014a.Mix-Headset                  36.60  1071.34   729.67   68.11       50.47        4.71           226.29            21.12    115.39     10.77
ES2014b.Mix-Headset                  25.20  2194.18  1709.23   77.90       68.04        3.10           338.44            15.42    146.51      6.68
ES2014c.Mix-Headset                  28.32  2286.79  1715.01   75.00       75.76        3.31           423.62            18.52    148.15      6.48
ES2014d.Mix-Headset                  48.14  2906.18  1606.31   55.27       99.20        3.41           550.03            18.93    749.84     25.80
IS1009a.Mix-Headset                  48.91   771.76   423.59   54.89       29.31        3.80           148.02            19.18    200.15     25.93
IS1009b.Mix-Headset                  19.26  2074.63  1712.89   82.56       37.75        1.82           288.50            13.91     73.24      3.53
IS1009c.Mix-Headset                  16.10  1680.33  1458.52   86.80       48.76        2.90           155.61             9.26     66.19      3.94
IS1009d.Mix-Headset                  37.43  1891.67  1244.28   65.78       60.61        3.20           323.68            17.11    323.71     17.11
TS3003a.Mix-Headset                  30.42  1209.16   869.21   71.89       27.88        2.31           243.57            20.14     96.38      7.97
TS3003b.Mix-Headset                  20.43  2011.69  1666.85   82.86       66.06        3.28           227.15            11.29    117.68      5.85
TS3003c.Mix-Headset                  21.80  2086.68  1683.40   80.67       51.67        2.48           282.95            13.56    120.32      5.77
TS3003d.Mix-Headset                  48.42  2394.09  1299.84   54.29       64.89        2.71           537.59            22.45    556.66     23.25
TS3007a.Mix-Headset                  37.25  1446.65   964.15   66.65       56.33        3.89           277.92            19.21    204.59     14.14
TS3007b.Mix-Headset                  19.06  2518.35  2109.07   83.75       70.65        2.81           282.46            11.22    126.82      5.04
TS3007c.Mix-Headset                  32.48  2902.49  2031.22   69.98       71.36        2.46           686.61            23.66    184.66      6.36
TS3007d.Mix-Headset                  39.55  3038.00  2044.97   67.31      208.42        6.86           715.14            23.54    277.89      9.15
TOTAL                                33.47 45854.38 31864.48   69.49     1356.14        2.96          9045.95            19.73   4943.95     10.78
```

This pipeline reaches 33.5% DER with no collar (or 15.3% DER with +/- 250ms collar and without scoring overlap regions).


## More options

For more options, see:

```bash
$ pyannote-pipeline --help
```

That's all folks!
