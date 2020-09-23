> The MIT License (MIT)
>
> Copyright (c) 2018-2020 CNRS
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

This tutorial assumes that you have already followed the [data preparation](../../data_preparation) tutorial, and teaches how to optimize a speech activity detection pipeline using `pyannote-pipeline` command line tool.

For simplicity, we will use a pretrained models for speech activity detection, speaker change detection, and speaker embeddings.

## Table of contents
- [Citation](#citation)
- [Raw scores extraction](#raw-scores-extraction)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speaker diarization, please cite the following paper:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

## Raw scores extraction
([↑up to table of contents](#table-of-contents))

We start by extracting raw scores/embeddings using the following pretrained models:

* `sad_ami` for speech activity detection
* `scd_ami` for speaker change detection
* `emb_ami` for speaker embedding 

```bash
$ export EXP_DIR=tutorials/pipelines/speaker_diarization

$ for SUBSET in developement test
 > do
 > for TASK in sad scd emb
 >  do
 >    pyannote-audio ${TASK} apply --step=0.1 --pretrained=${TASK}_ami --subset=${SUBSET} ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset    
 >  done
 > done
```

This tutorial relies on pretrained models available on `torch.hub` but you could (should?) obviously use a locally [trained](../../models/speech_activity_detection) or [fine-tuned](../../finetune) model.
In case you trained, validated and applied your own models by following the above tutorials, you may safely skip the corresponding `pyannote-audio ... apply` steps because you do not need to extract scores and/or embeddings again.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-pipeline` relies on a configuration file defining the experimental setup:

```bash
$ cat ${EXP_DIR}/config.yml
```
```yaml
pipeline:
  name: pyannote.audio.pipeline.speaker_diarization.SpeakerDiarization
  params:
    # replace {{EXP_DIR}} by its actual value
    sad_scores: {{EXP_DIR}}/sad_ami
    scd_scores: {{EXP_DIR}}/scd_ami
    embedding: {{EXP_DIR}}/emb_ami
    method: affinity_propagation

# one can freeze some of the hyper-parameters
# for instance, in this example, we are using
# hyper-parameters obtained in the speech 
# actitivy detection pipeline tutorial
freeze:
  speech_turn_segmentation:
    speech_activity_detection:
      min_duration_off: 0.6315121069334447
      min_duration_on: 0.0007366523493967721
      offset: 0.5727193137037349
      onset: 0.5842225805454029
      pad_offset: 0.0
      pad_onset: 0.0
```


If you are using any models that you trained, validated and applied locally [trained](../../models/speech_activity_detection) or [fine-tuned](../../finetune) models, and want to use your own set of scores, use their own paths instead. The example below uses pretrained embeddings but locally trained `sad` and `scd` scores:

```bash
$ cat ${EXP_DIR}/config.yml
```
```yaml
pipeline:
  name: pyannote.audio.pipeline.speaker_diarization.SpeakerDiarization
  params:
    sad_scores: /path/to/sad/experiment/train/{{TRAINING_SET}}/validate_detection_fscore/{{VALIDATION_SET}}/apply/{{BEST_EPOCH}}
    scd_scores: /path/to/scd/experiment/train/{{TRAINING_SET}}/validate_detection_fscore/{{VALIDATION_SET}}/apply/{{BEST_EPOCH}}
    # replace {{EXP_DIR}} by its actual value
    embedding: {{EXP_DIR}}/emb_ami
    method: affinity_propagation

# one can freeze some of the hyper-parameters
# for instance, in this example, we are using
# hyper-parameters obtained in the speech 
# actitivy detection pipeline tutorial
freeze:
  speech_turn_segmentation:
    speech_activity_detection:
      min_duration_off: 0.6315121069334447
      min_duration_on: 0.0007366523493967721
      offset: 0.5727193137037349
      onset: 0.5842225805454029
      pad_offset: 0.0
      pad_onset: 0.0
```

## Training
([↑up to table of contents](#table-of-contents))

The following command will run hyper-parameter optimization on the development subset of the AMI database. One can run it multiple times in parallel to speed things up.

```bash
$ pyannote-pipeline train --subset=development --forever ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```

Note that we use the `development` subset for optimizing the pipeline hper-parameters because the `train` subset has usually already been used for training the model itself.

This will create a bunch of files in `TRN_DIR`, including `params.yml` that contains the (so far) optimal parameters.


```bash
$ export TRN_DIR=${EXP_DIR}/train/AMI.SpeakerDiarization.MixHeadset.development
$ cat ${TRN_DIR}/params.yml
```
```yaml
loss: 0.3455305333795955
params:
  min_duration: 3.306092065580709
  speech_turn_assignment:
    closest_assignment:
      threshold: 0.8401481964056187
  speech_turn_clustering:
    clustering:
      damping: 0.6066098204003955
      preference: -2.9717704925136976
  speech_turn_segmentation:
    speaker_change_detection:
      alpha: 0.11115647156273972
      min_duration: 0.5283486365753665
    speech_activity_detection:
      min_duration_off: 0.6315121069334447
      min_duration_on: 0.0007366523493967721
      offset: 0.5727193137037349
      onset: 0.5842225805454029
      pad_offset: 0.0
      pad_onset: 0.0
```

The `loss:` value actually corresponds to the metric that is currently being optimized. For speaker diarization, the loss diarization error rate.

See `pyannote.audio.pipeline.speaker_diarization.SpeakerDiarization` docstring for details about the `params:` section.

Note that the actual content of your `params.yml` might vary because the optimisation process is not deterministic: the longer you wait, the better it gets. 

There is no easy way to decide if/when the optimization has converged to the optimal setting. The `pyannote-pipeline train` command will run forever, looking for a better set of hyper-parameters. 


## Application
([↑up to table of contents](#table-of-contents))

The optimized pipeline can then be applied on the `test` subset:

```bash
$ pyannote-pipeline apply --subset=test ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `${TRN_DIR}/apply/latest` subdirectory, including 
* `AMI.SpeakerDiarization.MixHeadset.test.rttm` that contains the actual output of the optimized pipeline
* `AMI.SpeakerDiarization.MixHeadset.test.eval` that provides an evaluation of the result (more or less equivalent to what you would get by using `pyannote.metrics` command line tool).

This pipeline reaches 32.2% DER with no collar:

```bash
$ pyannote-metrics diarization AMI.SpeakerDiarization.MixHeadset ${TRN_DIR}/apply/latest/AMI.SpeakerDiarization.MixHeadset.test.rttm
```
```
Diarization (collar = 0 ms)      diarization error rate    purity    coverage     total    correct      %    false alarm     %    missed detection      %    confusion      %
-----------------------------  ------------------------  --------  ----------  --------  ---------  -----  -------------  ----  ------------------  -----  -----------  -----
EN2002a.Mix-Headset                               43.21     59.42       58.64   2910.97    1691.18  58.10          37.92  1.30              992.73  34.10       227.06   7.80
EN2002b.Mix-Headset                               40.34     62.22       61.69   2173.78    1328.82  61.13          31.88  1.47              673.58  30.99       171.38   7.88
EN2002c.Mix-Headset                               31.56     70.58       70.56   3551.64    2467.64  69.48          36.93  1.04              955.66  26.91       128.33   3.61
EN2002d.Mix-Headset                               46.52     55.64       62.42   3042.98    1673.47  54.99          45.97  1.51             1089.70  35.81       279.81   9.20
ES2004a.Mix-Headset                               32.38     72.72       72.72   1051.71     737.47  70.12          26.30  2.50              260.22  24.74        54.01   5.14
ES2004b.Mix-Headset                               22.83     80.78       80.78   2403.80    1912.32  79.55          57.35  2.39              369.74  15.38       121.74   5.06
ES2004c.Mix-Headset                               25.36     78.18       78.18   2439.53    1895.33  77.69          74.36  3.05              392.17  16.08       152.03   6.23
ES2004d.Mix-Headset                               35.10     69.51       69.37   2258.48    1525.98  67.57          60.19  2.67              507.71  22.48       224.79   9.95
ES2014a.Mix-Headset                               38.06     70.57       77.58   1071.36     698.70  65.22          35.14  3.28              249.44  23.28       123.21  11.50
ES2014b.Mix-Headset                               24.90     80.45       80.39   2194.21    1699.32  77.45          51.55  2.35              356.14  16.23       138.75   6.32
ES2014c.Mix-Headset                               28.81     75.69       75.65   2286.85    1689.44  73.88          61.33  2.68              427.18  18.68       170.23   7.44
ES2014d.Mix-Headset                               29.89     75.73       75.50   2906.15    2116.52  72.83          79.09  2.72              561.41  19.32       228.22   7.85
IS1009a.Mix-Headset                               39.07     65.67       85.52    771.77     502.75  65.14          32.52  4.21              146.77  19.02       122.25  15.84
IS1009b.Mix-Headset                               23.38     78.82       78.82   2074.64    1629.99  78.57          40.45  1.95              284.50  13.71       160.14   7.72
IS1009c.Mix-Headset                               18.35     86.00       85.94   1680.33    1437.65  85.56          65.64  3.91              152.26   9.06        90.43   5.38
IS1009d.Mix-Headset                               36.34     68.28       76.39   1891.66    1277.60  67.54          73.37  3.88              312.00  16.49       302.07  15.97
TS3003a.Mix-Headset                               31.10     76.99       83.70   1209.19     861.34  71.23          28.18  2.33              240.27  19.87       107.58   8.90
TS3003b.Mix-Headset                               21.62     84.25       84.25   2011.71    1649.26  81.98          72.40  3.60              229.24  11.40       133.21   6.62
TS3003c.Mix-Headset                               23.82     83.40       83.39   2086.65    1655.75  79.35          66.12  3.17              287.59  13.78       143.30   6.87
TS3003d.Mix-Headset                               39.42     68.00       67.78   2394.10    1530.02  63.91          79.67  3.33              536.78  22.42       327.30  13.67
TS3007a.Mix-Headset                               38.59     67.35       83.73   1446.64     953.88  65.94          65.51  4.53              274.90  19.00       217.86  15.06
TS3007b.Mix-Headset                               20.67     82.52       82.51   2518.34    2066.54  82.06          68.72  2.73              277.84  11.03       173.96   6.91
TS3007c.Mix-Headset                               33.20     69.63       69.63   2902.52    2010.07  69.25          71.18  2.45              681.09  23.47       211.36   7.28
TS3007d.Mix-Headset                               44.10     63.69       63.69   3038.05    1928.17  63.47         229.80  7.56              709.90  23.37       399.99  13.17
TOTAL                                             32.24     72.22       73.84  52317.07   36939.21  70.61        1491.56  2.85            10968.84  20.97      4409.02   8.43
```

and 11.7% DER with +/- 250ms collar and without scoring overlap regions:

```bash
$ pyannote-metrics diarization --collar=0.5 --skip-overlap AMI.SpeakerDiarization.MixHeadset ${TRN_DIR}/apply/latest/AMI.SpeakerDiarization.MixHeadset.test.rttm
```
```
Diarization (collar = 500 ms, no overlap)      diarization error rate    purity    coverage     total    correct      %    false alarm      %    missed detection     %    confusion      %
-------------------------------------------  ------------------------  --------  ----------  --------  ---------  -----  -------------  -----  ------------------  ----  -----------  -----
EN2002a.Mix-Headset                                              9.60     92.56       61.94   1032.05     936.97  90.79           4.04   0.39               18.25  1.77        76.83   7.44
EN2002b.Mix-Headset                                              9.08     93.16       65.11    853.56     781.02  91.50           4.95   0.58               13.70  1.61        58.83   6.89
EN2002c.Mix-Headset                                              6.94     96.45       72.71   1641.68    1539.29  93.76          11.57   0.70               45.76  2.79        56.63   3.45
EN2002d.Mix-Headset                                             16.01     86.32       64.72   1006.27     855.91  85.06          10.77   1.07               14.46  1.44       135.90  13.51
ES2004a.Mix-Headset                                              9.90     95.82       75.44    539.48     495.14  91.78           9.09   1.68               22.74  4.21        21.60   4.00
ES2004b.Mix-Headset                                              6.86     95.85       82.73   1581.55    1494.85  94.52          21.82   1.38               21.96  1.39        64.75   4.09
ES2004c.Mix-Headset                                              7.82     94.55       80.23   1526.44    1435.10  94.02          27.97   1.83                8.60  0.56        82.74   5.42
ES2004d.Mix-Headset                                             13.18     90.49       72.49   1172.72    1032.94  88.08          14.83   1.26               29.32  2.50       110.46   9.42
ES2014a.Mix-Headset                                             24.63     86.20       78.71    688.23     541.13  78.63          22.43   3.26               60.47  8.79        86.63  12.59
ES2014b.Mix-Headset                                              9.39     94.74       82.55   1460.75    1335.75  91.44          12.20   0.84               50.90  3.48        74.10   5.07
ES2014c.Mix-Headset                                             10.78     93.09       77.66   1381.93    1255.55  90.85          22.63   1.64               33.14  2.40        93.24   6.75
ES2014d.Mix-Headset                                             12.28     92.46       78.01   1727.88    1538.56  89.04          22.85   1.32               62.36  3.61       126.97   7.35
IS1009a.Mix-Headset                                             23.88     80.11       87.01    425.01     338.36  79.61          14.86   3.50                2.18  0.51        84.47  19.88
IS1009b.Mix-Headset                                              6.40     94.48       81.23   1412.03    1330.36  94.22           8.68   0.62                3.94  0.28        77.72   5.50
IS1009c.Mix-Headset                                              5.50     96.73       87.56   1281.21    1235.24  96.41          24.52   1.91                4.07  0.32        41.90   3.27
IS1009d.Mix-Headset                                             17.64     84.94       78.42   1163.96     980.83  84.27          22.15   1.90                8.08  0.69       175.05  15.04
TS3003a.Mix-Headset                                             15.06     92.26       85.78    802.77     687.10  85.59           5.21   0.65               57.99  7.22        57.68   7.18
TS3003b.Mix-Headset                                              9.70     94.48       86.11   1504.39    1394.90  92.72          36.37   2.42               28.07  1.87        81.42   5.41
TS3003c.Mix-Headset                                             12.91     93.80       85.21   1556.34    1392.06  89.44          36.60   2.35               72.19  4.64        92.08   5.92
TS3003d.Mix-Headset                                             22.34     85.55       70.82   1353.46    1077.55  79.61          26.40   1.95               90.09  6.66       185.81  13.73
TS3007a.Mix-Headset                                             17.15     86.65       85.70    805.86     689.05  85.50          21.35   2.65               10.43  1.29       106.39  13.20
TS3007b.Mix-Headset                                              6.84     94.71       84.72   1810.40    1707.00  94.29          20.50   1.13                8.07  0.45        95.33   5.27
TS3007c.Mix-Headset                                              7.15     94.42       72.36   1483.67    1395.42  94.05          17.89   1.21                5.78  0.39        82.47   5.56
TS3007d.Mix-Headset                                             22.83     87.54       67.03   1392.85    1214.14  87.17         139.31  10.00                5.95  0.43       172.77  12.40
TOTAL                                                           11.75     92.29       76.45  29604.50   26684.24  90.14         559.00   1.89              678.50  2.29      2241.76   7.57
```


## More options

For more options, see:

```bash
$ pyannote-pipeline --help
```

That's all folks!
