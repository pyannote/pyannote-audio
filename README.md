# pyannote-audio-v2
Rewriting pyannote.audio from scratch


## CLI

```bash
pyannote-audio-train task=vad model=debug protocol=Debug.SpeakerDiarization.Debug
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
