# Frequently asked questions

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

1. Manually annotate dozens of conversations as precisely as possible.
2. Separate them into train (80%), development (10%) and test (10%) subsets.
3. Setup the data for use with [`pyannote.database`](https://github.com/pyannote/pyannote-database#speaker-diarization).
4. Follow [this recipe](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/adapting_pretrained_pipeline.ipynb).
5. Enjoy.
