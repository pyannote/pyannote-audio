## Frequently asked questions

## How does one capitalize and pronounce the name of this awesome library?

📝 Written in lower case: `pyannote.audio` (or `pyannote` if you are lazy).  Not `PyAnnote` nor `PyAnnotate` (*sic*).
📢 [Pronounced](https://www.howtopronounce.com/french/pianote) like the french verb *pianoter*.  *pi* like in **pi**ano, not *py* like in **py**thon.
🎹 *pianoter* means *to play the piano* (hence the logo 🤯).

## Can I use gated models (and pipelines) offline?

**Short answer**: yes, see [this tutorial](tutorials/applying_a_model.ipynb) for models and [that one](tutorials/applying_a_pipeline.ipynb) for pipelines.

**Long answer**: gating models and pipelines allows [me](https://herve.niderb.fr) to know a bit more about `pyannote.audio` user base and eventually help me write grant proposals to make `pyannote.audio` even better. So, please fill gating forms as precisely as possible.

For instance, before gating `pyannote/speaker-diarization`, I had no idea that so many people were relying on it in production. Hint: sponsors are more than welcome! Maintaining open source libraries is time consuming.

That being said, this whole authentication process does not prevent you from using official `pyannote.audio` models offline (i.e. without going through the authentication process in every `docker run ...` or whatever you are using in production): see [this tutorial](tutorials/applying_a_model.ipynb) for models and [that one](tutorials/applying_a_pipeline.ipynb) for pipelines.

## **[Pretrained pipelines](https://huggingface.co/models?other=pyannote-audio-pipeline) do not produce good results on my data. What can I do?**

1. [Annotate](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/prodigy.md) dozens of conversations manually and separate them into development and test subsets in [`pyannote.database`](https://github.com/pyannote/pyannote-database#speaker-diarization).
2. [Optimize the hyper-parameters](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/voice_activity_detection.ipynb) of the pretained pipeline using the development set. If performance is still not good enough, go to step 3.
3. Annotate hundreds of conversations manually and set them up as training subset in `pyannote.database`.
4. [Fine-tune](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb) the models (on which the pipeline relies) using the training set.
5. [Optimize the hyper-parameters](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/voice_activity_detection.ipynb) of the pipeline using the fine-tuned models using the development set. If performance is still not good enough, go back to step 3.
