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

# Applying pretrained models on your own data

This tutorial assumes that you have already followed the [data preparation](../../data_preparation) tutorial.

For the purpose of this tutorial, we use models available on `torch.hub` that were pretrained on `AMI` training subset:

```python
import torch
# speech activity detection model trained on AMI training set
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
# speaker change detection model trained on AMI training set
scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
# overlapped speech detection model trained on AMI training set
ovl = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami')
# speaker embedding model trained on AMI training set
emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
```

Note that models will run on GPU by default if one is available.  
Both device and batch size can be specified manually if needed: 
```python
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami', device='cpu', batch_size=128)
```

We will apply those pretrained models on the first file of the `AMI` test subset.

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

:warning: If you trained your own models (e.g. with [this](../../models/speech_activity_detection) tutorial) it is obviously possible to use it by providing the path to the validation directory to the `Pretrained` class:

```python
from pyannote.audio.features import Pretrained
sad = Pretrained(validate_dir='/path/to/validation/directory')
```

:warning: If you would like to test those models on your own data, you could do something like this (or [define your own protocol](../../data_preparation)). 


```python
# one can use their own file like this...
test_file = {'uri': 'filename', 'audio': '/path/to/your/filename.wav'}
```

Note that, in case of domain mismatch between your data and the `AMI` corpus, you might be better off [training your own models](../../models/speech_activity_detection) or [fine-tuning a pretrained one](../../finetune).

## Segmentation

### Speech activity detection

```python
# obtain raw SAD scores (as `pyannote.core.SlidingWindowFeature` instance)
sad_scores = sad(test_file)

# binarize raw SAD scores
# NOTE: both onset/offset values were tuned on AMI dataset.
# you might need to use different values for better results.
from pyannote.audio.utils.signal import Binarize
binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, 
                    min_duration_off=0.1, min_duration_on=0.1)

# speech regions (as `pyannote.core.Timeline` instance)
speech = binarize.apply(sad_scores, dimension=1)
```

### Speaker change detection

```python
# obtain raw SCD scores (as `pyannote.core.SlidingWindowFeature` instance)
scd_scores = scd(test_file)

# detect peaks and return speaker homogeneous segments 
# NOTE: both alpha/min_duration values were tuned on AMI dataset.
# you might need to use different values for better results.
from pyannote.audio.utils.signal import Peak
peak = Peak(alpha=0.10, min_duration=0.10, log_scale=True)

# speaker change point (as `pyannote.core.Timeline` instance)
partition = peak.apply(scd_scores, dimension=1)
```

### Overlapped speech detection

```python
# obtain raw OVL scores (as `pyannote.core.SlidingWindowFeature` instance)
ovl_scores = ovl(test_file)

# binarize raw OVL scores
# NOTE: both onset/offset values were tuned on AMI dataset.
# you might need to use different values for better results.
from pyannote.audio.utils.signal import Binarize
binarize = Binarize(offset=0.55, onset=0.55, log_scale=True, 
                    min_duration_off=0.1, min_duration_on=0.1)

# overlapped speech regions (as `pyannote.core.Timeline` instance)
overlap = binarize.apply(ovl_scores, dimension=1)
```

### Visualization

```python
# let's visualize SAD, SCD and OVL results using pyannote.core visualization API
import numpy as np
from matplotlib import pyplot as plt
from pyannote.core import Segment, notebook

# only plot one minute (between t=120s and t=180s)
notebook.crop = Segment(120, 180)

# helper function to make visualization prettier
from pyannote.core import SlidingWindowFeature
plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]), scores.sliding_window)

# create a figure with 8 rows with matplotlib
nrows = 8
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(nrows * 2)

# 1st row: reference annotation
notebook.plot_annotation(test_file['annotation'], ax=ax[0], time=False)
ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)

# 2nd row: SAD raw scores
notebook.plot_feature(plot_ready(sad_scores), ax=ax[1], time=False)
ax[1].text(notebook.crop.start + 0.5, 0.1, 'speech activity\ndetection scores', fontsize=14)
ax[1].set_ylim(-0.1, 1.1)

# 3rd row: SAD result
notebook.plot_timeline(speech, ax=ax[2], time=False)
ax[2].text(notebook.crop.start + 0.5, 0.1, 'speech activity detection', fontsize=14)

# 4th row: SCD raw scores
notebook.plot_feature(plot_ready(scd_scores), ax=ax[3], time=False)
ax[3].text(notebook.crop.start + 0.5, 0.1, 'speaker change\ndetection scores', fontsize=14)
ax[3].set_ylim(-0.1, 0.6)

# 5th row: SCD result
notebook.plot_timeline(partition, ax=ax[4], time=False)
ax[4].text(notebook.crop.start + 0.5, 0.1, 'speaker change detection', fontsize=14)

# 6th row: OVL raw scores
notebook.plot_feature(plot_ready(ovl_scores), ax=ax[5], time=False)
ax[5].text(notebook.crop.start + 0.5, 0.2, 'overlapped speech\ndetection scores', fontsize=14)
ax[5].set_ylim(-0.1, 1.1)

# 7th row: OVL result
notebook.plot_timeline(overlap, ax=ax[6], time=False)
ax[6].text(notebook.crop.start + 0.5, 0.1, 'overlapped speech detection', fontsize=14)

# 8th row: reference annotation
notebook.plot_annotation(test_file['annotation'], ax=ax[7], legend=False)
_ = ax[7].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)
```

![segmentation](segmentation.png)

### Speaker embedding

```python
# speech turns are simply the intersection of SAD and SCD
speech_turns = partition.crop(speech)
```

```python
# obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
embeddings = emb(test_file)

chunks = embeddings.sliding_window
print(f'Embeddings were extracted every {1000 * chunks.step:g}ms on {1000 * chunks.duration:g}ms-long windows.')
```

```python
# for the purpose of this tutorial, we only work of long (> 1s) speech turns
from pyannote.core import Timeline
long_turns = Timeline(segments=[s for s in speech_turns if s.duration > 2.])
```

Extracting embedding for a given speech turn is as easy as `embedding.crop(segment)`:

```python
X, Y = [], []
for segment in long_turns:
    # "strict" only keeps embedding strictly included in segment
    x = embeddings.crop(segment, mode='strict')
    # average speech turn embedding
    X.append(np.mean(x, axis=0))

    # keep track of speaker label (for later scatter plot)
    y = test_file['annotation'].argmax(segment)
    Y.append(y)

X = np.vstack(X)
_, y_true = np.unique(Y, return_inverse=True)
```

We can use *tSNE* to visualize (and later cluster, maybe?) embeddings.

```python
# apply tSNE on embeddings
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, metric="cosine")
X_2d = tsne.fit_transform(X)

# plot 
fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(5)
plt.scatter(*X_2d.T, c=y_true)
```

![tsne](tsne.png)

That's all folks!
