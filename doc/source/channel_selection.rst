Audio channel selection
=======================

Channel selection is controlled by the input file dictionary, consistently for
full-file loading, cropping, training, and inference. ``Audio`` only accepts a
target ``sample_rate``; it has no loader-wide channel setting.

.. list-table::
   :header-rows: 1

   * - ``file["channel"]``
     - Behavior
   * - Omitted, ``None``, ``"downmix"``, or ``Channel.DOWNMIX``
     - Average all channels before resampling.
   * - Non-negative integer
     - Select that zero-based channel before resampling.
   * - ``"all"`` or ``Channel.ALL``
     - Preserve all channels in their original order and resample each.
   * - ``"random"`` or ``Channel.RANDOM``
     - Select one channel independently on each load or crop, then resample.

``Channel`` is a string enum in ``pyannote.audio.core.io``. Invalid policies and
out-of-range channel indices raise ``ValueError``. Random selection uses Python's
random generator and is reproducible when that generator is seeded.

For example::

    from pyannote.audio.core.io import Audio, Channel

    loader = Audio(sample_rate=16000)
    waveform, sample_rate = loader({"audio": "recording.wav", "channel": Channel.ALL})

Training and validation
-----------------------

Protocols may specify any policy for training. Validation rejects ``"random"``
so that channel selection does not introduce variation between evaluations.
Models declare their required ``num_channels`` and reject mismatched inputs;
they do not silently downmix in their forward pass. A multi-channel architecture
must explicitly support its declared channel count. Batches require the same
channel count for every example.

Task caches store integer channel indices and private sentinel values. Existing
integer caches remain compatible. Old caches that encoded string policies as
category indices must be rebuilt; loading such a cache raises a clear error.

Preloading and pipeline stages
------------------------------

``Pipeline.prepare_one(..., preload=True)`` and the ``Waveform`` preprocessor
load raw channels and retain the file's selection policy. Selection happens in
each consumer, preserving the same behavior with and without preloading.
In particular, preloading does not choose a random channel for the whole file.
To share one random channel across multiple stages, choose an integer once and
put it in the file dictionary.

The diarization and separation pipelines accept ``embedding_channel`` to override
the file policy for embedding extraction only. Its default, ``None``, inherits
the file policy. For example, a multi-channel segmentation model can receive
``channel: all`` while ``embedding_channel="downmix"`` feeds mono embeddings.
This override does not modify the input dictionary or the segmentation policy.

Migrating from ``mono``
-----------------------

This is a breaking API change. ``Audio(mono=...)`` is no longer accepted.

* Replace ``Audio(mono="downmix")`` with ``Audio()``. Use ``channel: downmix``
  explicitly when desired.
* Replace ``Audio(mono="random")`` with ``Audio()`` and ``channel: random`` on
  training files.
* Bare ``Audio()`` and ``Audio(mono=None)`` previously preserved all channels.
  To retain that behavior, use ``Audio()`` with ``channel: all``.
* ``model.audio.mono`` no longer controls loading. Configure file dictionaries or
  protocols instead.

Default model training and inference continue to downmix when no policy is
provided. Raw waveform preprocessors intentionally preserve all channels for
later consumers.
