> The MIT License (MIT)
>
> Copyright (c) 2020- CNRS
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

# Applying pretrained pipelines on your own data

This tutorial assumes that you have already followed the [data preparation](../../data_preparation) tutorial.

For the purpose of this tutorial, we use a speaker diarization pipeline available on `torch.hub` that was pretrained on `AMI` training subset:

```python
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia_ami')
```

We will apply this pretrained pipeline on the first file of the `AMI` test subset.

```python
# ... or use a file provided by a pyannote.database protocol
# in this example, we are using AMI first test file.
from pyannote.database import get_protocol
from pyannote.database import FileFinder
preprocessors = {'audio': FileFinder()}
protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
                        preprocessors=preprocessors)
test_file = next(protocol.test())
```

:warning: If you would like to test this pipeline on your own data, you could do something like this (or [define your own protocol](../../data_preparation)). 

```python
# one can use their own file like this...
test_file = {'uri': 'filename', 'audio': '/path/to/your/filename.wav'}
```

Note that, in case of domain mismatch between your data and the `AMI` corpus, you might be better off [training your own models](../../models) or [fine-tuning a pretrained one](../../finetune), and [tuning your own pipeline](../../pipelines).

## Diarization

```python
diarization = pipeline(test_file)
```

### Visualization

```python
# let's visualize the diarization output using pyannote.core visualization API
from matplotlib import pyplot as plt
from pyannote.core import Segment, notebook

# only plot one minute (between t=120s and t=180s)
notebook.crop = Segment(120, 180)

# create a figure with 6 rows with matplotlib
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(nrows * 2)

# 1st row: reference annotation
notebook.plot_annotation(test_file['annotation'], ax=ax[0], time=False)
ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)

# 2nd row: pipeline output
notebook.plot_annotation(diarization, ax=ax[1], time=False)
ax[1].text(notebook.crop.start + 0.5, 0.1, 'hypothesis', fontsize=14)
```

![diarization](diarization.png)

That's all folks!
