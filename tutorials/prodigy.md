# Annotating your own data with Prodigy

Manually annotating audio data is time consuming.  
`pyannote.audio` comes with a bunch of [Prodigy](https://prodi.gy) recipes that should speed things up a bit. 

## Pre-annotate with a pretrained pipeline

`pyannote.audio` recipe will stream in `.wav` files in chunks and apply [a pretrained pipeline](https://huggingface.co/models?other=pyannote-audio-pipeline). You can then adjust the regions manually if needed.

```bash
prodigy pyannote.audio dataset /path/to/audio/directory pyannote/speaker-segmentation
```

![pyannote.audio screenshot](./assets/prodigy-pyannote.audio.png)

## Merge multiple annotations

Work on `pyannote.review` in progress...

## Compare multiple annotations

Work on `pyannote.diff` in progress...
