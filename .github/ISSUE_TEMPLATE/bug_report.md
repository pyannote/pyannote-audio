---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
```bash
$ pyannote-audio sad train ...
$ pyannote-audio sad validate ...
```

**Content of `config.yml`**

```yaml
task:
   name: SpeechActivityDetection
   params:
      duration: 2.0
      batch_size: 8

feature_extraction:
   name: RawAudio
   params:
      sample_rate: 16000

architecture:
   name: pyannote.audio.models.PyanNet
   params:
      rnn:
         unit: LSTM
         hidden_size: 16
         num_layers: 1
         bidirectional: True
      ff:
         hidden_size: [16]

scheduler:
   name: ConstantScheduler
   params:
      learning_rate: 0.01
```

**pyannote environment**

```bash
$ pip freeze | grep pyannote
pyannote.audio==2.0.a
pyannote.core==3.7.1
pyannote.database==3.0.1
pyannote.db.musan==0.1.3
pyannote.db.voxceleb==1.0.1
pyannote.metrics==2.3
pyannote.pipeline==1.5
```

**Additional context**
Add any other context about the problem here.
