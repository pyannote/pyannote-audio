# Rewriting pyannote.audio from scratch

This is alpha release of upcoming `pyannote.audio` 2.0 for which it has been decided to rewrite almost everything from scratch to benefit from `pytorch-lightning` framework.

## Installation

Until a proper release is available on PyPI, install from the `develop` branch:

```bash
git clone https://github.com/pyannote/pyannote-audio.git
cd pyannote-audio
git checkout develop
pip install .
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
from pyannote.audio.core.inference import Inference
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
from pyannote.audio.core.model import load_from_checkpoint
model = load_from_checkpoint('hbredin/VoiceActivityDetection-PyanNet-DIHARD')
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
