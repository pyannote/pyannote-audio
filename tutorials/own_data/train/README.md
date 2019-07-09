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
> AUTHORS
> Hervé Bredin - http://herve.niderb.fr

# Training on your own dataset with `pyannote.audio`

In this tutorial, you will learn how to setup your own dataset so that models can be trained on it. We will call this dataset `YourDataset` in the rest of this tutorial. 


## Audio files

Though `pyannote.audio` supports other file formats (it is based on [SoundFile](https://pypi.org/project/SoundFile/)), let us assume that `YourDataset` contains 3 WAV files:

```
/path/to/your/dataset/audio/file1.wav
/path/to/your/dataset/audio/file2.wav
/path/to/your/dataset/audio/file3.wav
```

## Reference files

Your dataset needs to come with annotations to be of any use for training, in the form of RTTM and UEM files.

*"who speaks when"* annotations should be provided using the RTTM file format. Each line in this file must follow the following convention:

```
SPEAKER {uri} 1 {start} {duration} <NA> <NA> {identifier} <NA> <NA>
```

where `{uri}` stands for "unique resource identifier" (think of it as the filename), `{start}` is the start time (elapsed time since the beginning of the file, in seconds) of the speech turn, `{duration}` is its duration (in seconds) and `{identifier}` is the unique speaker identifier.

Here what it would look like for `YourDataset`:

```bash
$ cat /path/to/your/dataset/train.rttm
SPEAKER file1 1 0.130 3.880 <NA> <NA> alice <NA> <NA>
SPEAKER file1 1 4.790 0.960 <NA> <NA> alice <NA> <NA>
SPEAKER file1 1 6.190 0.910 <NA> <NA> bob <NA> <NA>
SPEAKER file1 1 7.670 2.340 <NA> <NA> alice <NA> <NA>
SPEAKER file1 1 10.830 2.400 <NA> <NA> carol <NA> <NA>
SPEAKER file1 1 13.670 3.430 <NA> <NA> carol <NA> <NA>
SPEAKER file2 1 17.900 2.210 <NA> <NA> john <NA> <NA>
SPEAKER file2 1 20.370 0.760 <NA> <NA> jack <NA> <NA>
SPEAKER file2 1 21.560 3.410 <NA> <NA> john <NA> <NA>
SPEAKER file3 1 25.500 3.410 <NA> <NA> hugh <NA> <NA>
```

It is possible that only parts of your files are annotated.
This is the role of the UEM file: telling `pyannote-audio` which part were actually annotated. 

If you do not provide this file, `pyannote-audio` assumes that the whole file was annotated and therefore everything that is outside of a speech turn is considered non-speech. 

If you do provide this file, `pyannote-audio` will only consider as non-speech those regions that are within the limits defined in the UEM file.

Each line in this file must follow the following convention:

```
{uri} 1 {start} {end}
```

Here is what it might look like for `YourDataset`

```bash
$ cat /path/to/your/dataset/train.uem
file1 1 0.000 120.0
file1 1 130.0 240.0
file2 1 0.000 300.0
file3 1 60.0 300.0
```

## Configuration file

Once everything is ready, you can update (or create if it does not exist) file `/path/to/database.yml` like this:

```yaml
Databases:
  YourDataset: /path/to/your/dataset/audio/{uri}.wav

Protocols:
  YourDataset:
    SpeakerDiarization:
      YourProtocol:
        train:
          annotation: /path/to/your/dataset/train.rttm
          annotated: /path/to/your/dataset/train.uem
```

... and tell `pyannote.database` about this file:

```bash
$ export PYANNOTE_DATABASE_CONFIG=/path/to/database.yml
```

Congratulations: you have just defined a new `pyannote.database` protocol (called `YourDataset.SpeakerDiarization.YourProtocol`) that can be used in `pyannote.audio`. 

All you have to do now is to replace `AMI.SpeakerDiarization.MixHeadset` by `YourDataset.SpeakerDiarization.YourProtocol` in the tutorials explaining how to train models...

Note that you should probably add a `development` set for validating the models (and optionally a `test` set for proper evaluation):

```yaml
Databases:
  YourDataset: /path/to/your/dataset/audio/{uri}.wav

Protocols:
  YourDataset:
    SpeakerDiarization:
      YourProtocol:
        train:
          annotation: /path/to/your/dataset/train.rttm
          annotated: /path/to/your/dataset/train.uem
        development:
          annotation: /path/to/your/dataset/development.rttm
          annotated: /path/to/your/dataset/development.uem
        test:
          annotation: /path/to/your/dataset/test.rttm
          annotated: /path/to/your/dataset/test.uem
```

That's all folks!
