> The MIT License (MIT)
>
> Copyright (c) 2017-2020 CNRS
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

# Fine-tuning a pretrained model to your own data

This tutorial assumes that you have already followed the [data preparation](../data_preparation) tutorial and teaches how to fine-tune a pretrained model to your own data.

The list of pretrained models available in `pyannote.audio` can be obtained with:

```python
import torch
torch.hub.list('pyannote/pyannote-audio')
```

More precisely, we will fine-tune a speech activity detection model (which was originally trained on the DIHARD dataset) to the [AMI](http://groups.inf.ed.ac.uk/ami/corpus) dataset.

It is recommended to (at least) read [this](../models/speech_activity_detection) tutorial first, especially the part related to the configuration file.

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Fine tuning](#fine-tuning)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio`, please cite the following paper:

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

## Configuration
([↑up to table of contents](#table-of-contents))

Because pretrained models come with their own configuration file, no configuration file is needed to fine-tune a model. 

```bash
$ export EXP_DIR=tutorials/finetune
$ cat ${EXP_DIR}/config.yml
cat: config.yml: No such file or directory
```

Note that you can still choose to provide one (possibly partial) configuration file to override sections of the pretrained model configuration file.

```yaml
# this (partial) configuration file overrides the batch size
task:
   name: SpeechActivityDetection
   params:
      batch_size: 16
```

## Fine tuning
([↑up to table of contents](#table-of-contents))

The following command will fine-tune the `sad_dihard` pretrained model using the training subset of AMI database for 5 epochs:

```bash
$ pyannote-audio sad train --pretrained=sad_dihard --subset=train --to=5 --parallel=4 ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```

Your model is now fine-tuned! Note that a configuration file has been created for you in `EXP_DIR`.

## More options
([↑up to table of contents](#table-of-contents))

You may also use your own pretrained model. Simply provide the absolute path to the model checkpoint instead of `sad_dihard`:

```bash
$ pyannote-audio sad train \
  --pretrained=/path/to/train/YourDataset.SpeakerDiarization.YourProtocol.train/weights/0020.pt \
  --subset=train --to=5 --parallel=4 \
  ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```

Go check [this](../models/speech_activity_detection) tutorial to learn how to validate and use your brand new model.

That's all folks!
