---
title: "Can I use gated models (and pipelines) offline?"
alt_titles:
  - "Why does one need to authenticate to access the pretrained models?"
  - "Can I use pyannote.audio pretrained pipelines without the Hugginface token?"
  - "How can I solve the permission issue?"
---

**Short answer**: yes, see [this tutorial](tutorials/applying_a_model.ipynb) for models and [that one](tutorials/applying_a_pipeline.ipynb) for pipelines.

**Long answer**: gating models and pipelines allows [me](https://herve.niderb.fr) to know a bit more about `pyannote.audio` user base and eventually help me write grant proposals to make `pyannote.audio` even better. So, please fill gating forms as precisely as possible.

For instance, before gating `pyannote/speaker-diarization`, I had no idea that so many people were relying on it in production. Hint: sponsors are more than welcome! Maintaining open source libraries is time consuming.

That being said, this whole authentication process does not prevent you from using official `pyannote.audio` models offline (i.e. without going through the authentication process in every `docker run ...` or whatever you are using in production): see [this tutorial](tutorials/applying_a_model.ipynb) for models and [that one](tutorials/applying_a_pipeline.ipynb) for pipelines.
