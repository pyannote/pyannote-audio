# Neural speaker diarization with pyannote.audio

This is the development branch of upcoming `pyannote.audio` 2.0 for which it has been decided to rewrite almost everything from scratch.  Highlights of this upcoming release will be:

- a much smaller and cleaner codebase
- Python-first API (the *good old* pyannote-audio CLI will still be available, though)
- multi-GPU and TPU training thanks to [pytorch-lightning](https://pytorchlightning.ai/)
- data augmentation with [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
- [huggingface](https://huggingface.co) model hosting
- [prodigy](https://prodi.gy) recipes for audio annotations
- online [demo](https://share.streamlit.io/pyannote/pyannote-audio-demo/main/app.py) based on [streamlit](https://www.streamlit.io)

## Installation

```bash
conda create -n pyannote python=3.8.5
conda activate pyannote

# pyannote.audio relies on torchaudio's soundfile backend, itself relying
# on libsndfile, sometimes tricky to install. This seems to work fine but
# is provided with no guarantee of success:
conda install numpy cffi
conda install libsndfile=1.0.28 -c conda-forge

# until a proper release of pyannote.audio 2.x is available on PyPI,
# install from the `develop` branch:
pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
```

## pyannote.audio 101

*For now, this is the closest you can get to an actual documentation.*

Experimental protocol is reproducible thanks to [`pyannote.database`](https://github.com/pyannote/pyannote-database).
*Here, we use the [AMI](https://github.com/BUTSpeechFIT/AMI-diarization-setup) "only_words" speaker diarization protocol.*

```python
from pyannote.database import get_protocol
ami = get_protocol('AMI.SpeakerDiarization.only_words')
```

Data augmentation is supported via [`torch-audiomentations`](https://github.com/asteroid-team/torch-audiomentations).

```python
from torch_audiomentations import Compose, ApplyImpulseResponse, AddBackgroundNoise
augmentation = Compose(transforms=[ApplyImpulseResponse(...),
                                   AddBackgroundNoise(...)])
```

A growing collection of tasks can be addressed.
*Here, we address speaker segmentation.*

```python
from pyannote.audio.tasks import Segmentation
seg = Segmentation(ami, augmentation=augmentation)
```

A growing collection of model architecture can be used.
*Here, we use the PyanNet (sincnet + LSTM) architecture.*

```python
from pyannote.audio.models.segmentation import PyanNet
model = PyanNet(task=seg)
```

We benefit from all the nice things that [`pytorch-lightning`](https://www.pytorchlightning.ai/) has to offer:  distributed (GPU & TPU) training, model checkpointing, logging, etc.
*In this example, we don't really use any of this...*

```python
from pytorch_lightning import Trainer
trainer = Trainer()
trainer.fit(model)
```

Predictions are obtained by wrapping the model into the `Inference` engine.

```python
from pyannote.audio import Inference
inference = Inference(model)
predictions = inference('audio.wav')
```

Pretrained models can be shared on [Huggingface.co](https://huggingface.co) model hub.
*Here, we download and use a [pretrained](https://huggingface.co/hbredin/VoiceActivityDetection-PyanNet-DIHARD) voice activity detection model.*

```python
inference = Inference('hbredin/VoiceActivityDetection-PyanNet-DIHARD')
predictions = inference('audio.wav')
```

Fine-tuning is as easy as setting the `task` attribute, freezing early layers and training.
*Here, we fine-tune on AMI dataset a voice activity detection model pretrained on DIHARD dataset.*

```python
from pyannote.audio import Model
model = Model.from_pretrained('hbredin/VoiceActivityDetection-PyanNet-DIHARD')
model.task = VoiceActivityDetection(ami)
model.freeze_up_to('sincnet')
trainer.fit(model)
```

Transfer learning is also supported out of the box.
*Here, we do transfer learning from voice activity detection to overlapped speech detection.*

```python
from pyannote.audio.tasks import OverlappedSpeechDetection
osd = OverlappedSpeechDetection(ami)
model.task = osd
trainer.fit(model)
```

Default optimizer (`Adam` with default parameters) is automatically set up for you.  Customizing optimizer (and scheduler) requires overriding [`model.configure_optimizers`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers) method:

```python
from types import MethodType
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
def configure_optimizers(self):
    return {"optimizer": SGD(self.parameters()),
            "lr_scheduler": ExponentialLR(optimizer, 0.9)}
model.configure_optimizers = MethodType(configure_optimizers, model)
trainer.fit(model)
```

## Contributing

The commands below will setup pre-commit hooks and packages needed for developing the `pyannote.audio` library.

```bash
pip install -e .[dev,testing]
pre-commit install
```

## Testing

Tests rely on a set of debugging files available in [`test/data`](test/data) directory.
Set `PYANNOTE_DATABASE_CONFIG` environment variable to `test/data/database.yml` before running tests:

```bash
PYANNOTE_DATABASE_CONFIG=tests/data/database.yml pytest
```
