> The MIT License (MIT)
>
> Copyright (c) 2019-2020 CNRS
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


# Preparing your own dataset for `pyannote.audio`

In this tutorial, you will learn how to setup your own audio dataset for use with `pyannote.audio`.  
For illustration purposes, we will use the freely available [AMI corpus](http://groups.inf.ed.ac.uk/ami/corpus):

```bibtex
@article{Carletta2007,  
  Title = {{Unleashing the killer corpus: experiences in creating the multi-everything AMI Meeting Corpus}},
  Author = {Carletta, Jean},
  Journal = {Language Resources and Evaluation},
  Volume = {41},
  Number = {2},
  Year = {2007},
}
```

Start by cloning the `pyannote.audio` repository:

```bash
$ git clone https://github.com/pyannote/pyannote-audio.git
$ cd pyannote-audio
$ export TUTORIAL_DIR="$PWD/tutorials/data_preparation"
```

## Audio files

[Download](http://groups.inf.ed.ac.uk/ami/download/) the `Headset mix` subset of `AMI`.  For convenience, we provide a [script](./download_ami.sh) that does it for you: 

```bash
$ export DOWNLOAD_TO=/path/to/where/you/want/to/download/ami/database
$ mkdir -p ${DOWNLOAD_TO}
$ sh ${TUTORIAL_DIR}/download_ami.sh ${DOWNLOAD_TO}
```

This script also *fixes* some of the files from the dataset that are unreadable with `scipy` because of wrongly formatted wav chunks. The audio files are therefore not exactly the same as the original ones. You should end up with a collection of `wav` files in the `${DOWNLOAD_TO}/amicorpus` directory:

```bash
$ cd ${DOWNLOAD_TO}/amicorpus
$ find . | grep wav | sort | head -n 5
./EN2001a/audio/EN2001a.Mix-Headset.wav
./EN2001b/audio/EN2001b.Mix-Headset.wav
./EN2001d/audio/EN2001d.Mix-Headset.wav
./EN2001e/audio/EN2001e.Mix-Headset.wav
./EN2002a/audio/EN2002a.Mix-Headset.wav
```

Most `pyannote.audio` tutorials rely on noise extracted from the [MUSAN corpus](http://www.openslr.org/resources/17/musan.tar.gz) for data augmentation.  
For convenience, we also provide a [script](./download_musan.sh) that downloads it for you.

```bash
$ sh ${TUTORIAL_DIR}/download_musan.sh ${DOWNLOAD_TO}
```

You should end up with a collection of `wav` files in the `${DOWNLOAD_TO}/musan` directory:

```bash
$ cd ${DOWNLOAD_TO}/musan
$ find . | grep wav | sort | head -n 5
./music/fma/music-fma-0000.wav
./music/fma/music-fma-0001.wav
./music/fma/music-fma-0002.wav
./music/fma/music-fma-0003.wav
./music/fma/music-fma-0004.wav
```

`pyannote.audio` relies on `pyannote.database` that itself relies on a configuration file that indicates where files are located. 
For convenience, we also provide [such a file](./database.yml) that needs to be copied at the root of the directory that contains the datasets.

```bash
$ cp ${TUTORIAL}/database.yml ${DOWNLOAD_TO}
```

See [`pyannote.database` documentation](https://github.com/pyannote/pyannote-database#preprocessors) for other possible locations for `database.yml`.

## Reference files

Your dataset needs to come with annotations to be of any use for training, in the form of RTTM and UEM files.

*"who speaks when"* annotations should be provided using the RTTM file format. Each line in this file must follow the following convention:

```
SPEAKER {uri} 1 {start} {duration} <NA> <NA> {identifier} <NA> <NA>
```

where 
* `{uri}` stands for "unique resource identifier" (think of it as the filename), 
* `{start}` is the start time (elapsed time since the beginning of the file, in seconds) of the speech turn, 
* `{duration}` is its duration (in seconds),
* `{identifier}` is the unique speaker identifier.

For convenience, we also provide RTTM reference files for AMI dataset.
Here what it looks like for the training subset:

```bash
$ head -n 10 ${TUTORIAL_DIR}/AMI/MixHeadset.train.rttm
SPEAKER ES2002b.Mix-Headset 1 14.4270 1.3310 <NA> <NA> FEE005 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 16.9420 1.0360 <NA> <NA> FEE005 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 19.4630 1.0600 <NA> <NA> FEE005 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 23.0420 41.1690 <NA> <NA> FEE005 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 27.3530 1.2570 <NA> <NA> MEE008 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 30.4780 0.4810 <NA> <NA> MEE007 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 30.4970 3.6660 <NA> <NA> MEE008 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 36.4310 2.5000 <NA> <NA> MEE008 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 100.5660 49.9540 <NA> <NA> FEE005 <NA> <NA>
SPEAKER ES2002b.Mix-Headset 1 116.6600 0.4940 <NA> <NA> MEE008 <NA> <NA>
```

:warning: It is possible that only parts of your files are annotated. This is the role of the UEM file: to tell `pyannote-audio` which part were actually annotated.  
* If you do not provide this file, `pyannote-audio` assumes that the whole file was annotated and therefore everything that is outside of a speech turn is considered non-speech.
* If you do provide this file, `pyannote-audio` will only consider as non-speech those regions that are within the limits defined in the UEM file.

Each line in this file must follow the following convention:

```
{uri} 1 {start} {end}
```

Here is what it looks like for `AMI` training subset

```bash
$ head -n 10 ${TUTORIAL_DIR}/AMI/MixHeadset.train.uem
EN2001a.Mix-Headset 1 0.000 5250.240063
EN2001b.Mix-Headset 1 0.000 3451.882687
EN2001d.Mix-Headset 1 0.000 3546.378688
EN2001e.Mix-Headset 1 0.000 4069.738688
EN2003a.Mix-Headset 1 0.000 2240.277375
EN2004a.Mix-Headset 1 0.000 3445.674687
EN2005a.Mix-Headset 1 0.000 5415.234688
EN2006a.Mix-Headset 1 0.000 3525.493375
EN2006b.Mix-Headset 1 0.000 3052.258688
EN2009b.Mix-Headset 1 0.000 2474.282688
```

:warning: It is recommended to always provide a `UEM` file (even if it covers the whole duration of each file).

## Configuration file

`pyannote.database` relies on the same configuration file to define train/development/test splits. Once again, we have prepared this for you, following the official split defined [here](http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).

All you have to do is to copy the annotation files at the root of the directory that contains the datasets.

```bash
# replace TUTORIAL_DIR placeholder with actual location
$ cp -r ${TUTORIAL_DIR}/AMI ${DOWNLOAD_TO}
```

It should look like this:

```bash
# have a quick look at what it looks like
$ cat ${DOWNLOAD_TO}/database.yml
```

```yaml
Databases:
   AMI: ./amicorpus/*/audio/{uri}.wav
   MUSAN: ./musan/{uri}.wav

Protocols:
   AMI:
      SpeakerDiarization:
         MixHeadset:
           train:
              annotation: ./AMI/MixHeadset.train.rttm
              annotated: ./AMI/MixHeadset.train.uem
           development:
              annotation: ./AMI/MixHeadset.development.rttm
              annotated: ./AMI/MixHeadset.development.uem
           test:
              annotation: ./AMI/MixHeadset.test.rttm
              annotated: ./AMI/MixHeadset.test.uem
```

You might want to have a look at `pyannote.database` [documentation](http://github.com/pyannote/pyannote-database) to learn more about other features provided by `pyannote.database` and the syntax of its configuration file.

Did you notice that we did not provide any `MUSAN` entry in the `Protocols` section. This is because another (more flexible) way of providing annotations is possible through the provision of `pyannote.database` [plugins](https://github.com/pyannote/pyannote-db-template). And it happens that `MUSAN` has one called `pyannote.db.musan`. Simply install it with the following command line:

```bash
# install MUSAN plugin
$ pip install pyannote.db.musan
```

The final step is to tell `pyannote.database` about the location of the configuration file:

```bash
# tell pyannote.database about the location of the configuration file
$ export PYANNOTE_DATABASE_CONFIG=${DOWNLOAD_TO}/database.yml
```

**Congratulations:** you have just defined a new `pyannote.database` protocol called `AMI.SpeakerDiarization.MixHeadset` that will be used in other tutorials.

That's all folks!
