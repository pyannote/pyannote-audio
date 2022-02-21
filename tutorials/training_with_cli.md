# Training with `pyannote-audio-train` command line tool

*This tutorial is a very preliminary draft. Expect hiccups and missing details.*

`pyannote.audio` provides a command line tool called `pyannote-audio-train` 
that allows to train models directly from your terminal. It relies on extra
dependencies installed using th `[cli]` suffix:

```bash
pip install pyannote-audio[cli]
```


## TL;DR

Calling the following command would train a PyanNet model for voice activity
detection on the AMI corpus...

```bash
pyannote-audio-train \
    model=PyanNet \
    task=VoiceActivityDetection \
    protocol=AMI.SpeakerDiarization.only_words
```

... which is more or less equivalent to running the following Python script:

```python
from pyannote.audio.tasks import VoiceActivityDetection
from pyannote.audio.models.segmentation import PyanNet
from pyannote.database import get_protocol
from pytorch_lightning import Trainer

protocol = get_protocol("AMI.SpeakerDiarization.only_words")
task = VoiceActivityDetection(protocol)
model = PyanNet(task=task)
trainer = Trainer()
trainer.fit(model)
```

## Hydra-based configuration

`pyannote-audio-train` relies on [`Hydra`](https://hydra.cc) to configure the 
training process. Adding `--cfg job` option to the previous command will let 
you know about the actual configuration used for training:


```bash
pyannote-audio-train --cfg job \
    model=PyanNet \
    task=VoiceActivityDetection \
    protocol=AMI.SpeakerDiarization.only_words
```

```yaml
task:
  _target_: pyannote.audio.tasks.VoiceActivityDetection
  duration: 3.0
  warm_up: 0.0
  balance: null
  weight: null
  batch_size: 32
  num_workers: null
  pin_memory: false
[...]
```

To change the duration of audio chunks used for training to 2 seconds, you would do

```bash
pyannote-audio-train \
    model=PyanNet \
    task=VoiceActivityDetection task.duration=2.0 \
    protocol=AMI.SpeakerDiarization.only_words
```

You get the idea...

### Configuring data augmentation

Create a YAML file that can be loaded by [torch_audiomentations's from_dict](https://github.com/asteroid-team/torch-audiomentations/blob/cb7b3ec10ee1c4951a04d08bb94294ce28a971de/torch_audiomentations/utils/config.py#L14-L39) utility function:

```bash
cat /path/to/custom_config/augmentation/background_noise.yaml
```

```yaml
# @package _group_
transform: Compose
params:
  shuffle: False
  transforms:
    - transform: AddBackgroundNoise
      params:
        background_paths: /path/to/directory/containing/background/audio
        min_snr_in_db: 5.
        max_snr_in_db: 15.
        mode: per_example
        p: 0.9
```

```bash
pyannote-audio-train \
    --config-dir /path/to/custom_config \
    model=PyanNet \
    task=VoiceActivityDetection task.duration=2.0 \
    protocol=AMI.SpeakerDiarization.only_words \
    +augmentation=background_noise
```

## Training on a Slurm cluster

As described in `Hydra`'s documentation, `hydra-submitit-launcher` allows
to launch multiple jobs (e.g. to perform a grid search on some hyper-parameters).


```bash
pip install hydra-submitit-launcher --upgrade
```

Here, we launch a grid of (3 x 2 =) six different jobs:
* 2, 3, or 4 LSTM layers
* mono-directional or bidirectional LSTMs

```bash
pyannote-audio-train 
    --multirun hydra/launcher=submitit_slurm \
    model=PyanNet +model.lstm.num_layers=2,3,4 +model.lstm.bidirectional=true,false \
    task=VoiceActivityDetection \
    protocol=AMI.SpeakerDiarization.only_words
```

Known bugs: pytorch-lightning + hydra-submitit + multi-GPU do not play well together ([here](https://github.com/PyTorchLightning/pytorch-lightning/issues/2727), [here](https://github.com/PyTorchLightning/pytorch-lightning/issues/11300), and [here](https://github.com/PyTorchLightning/pytorch-lightning/pull/11617))

### Training on Jean Zay cluster

```bash
+hydra.launcher.additional_parameters.account=eie@gpu   # --account option
hydra.launcher.qos=qos_gpu-dev                          # QOS
hydra.launcher.gpus_per_task=1                          # number of GPUs
hydra.launcher.cpus_per_gpu=10                          # number of CPUS per GPUs (10 is )
hydra.launcher.timeout_min=120                          # --time option (in minutes)
task.duration=2,5,10 hydra.sweep.subdir=\${task.duration}s_chunks
```
