Annotation versions in task caches
=================================

``Task.prepare_data`` groups protocol entries sharing ``(database, uri)`` into
one recording. Word-level and turn-level annotations of that recording therefore
share a ``file_id``, an audio entry, and a file-scoped speaker-label mapping.
Their segments occupy one contiguous slice in ``audio-segments-ids`` even when
the protocol yields versions in different places.

Versions must agree on audio path, channel selection, and speaker scope. A
recording cannot appear in both training and development subsets. Speaker labels
must identify the same people across versions. Other recording metadata are
taken from the first entry; top-level ``level`` and ``version`` are not stored as
audio metadata. Use distinct database names or URIs for different audio conditions.

Levels and versions
-------------------

The level is read from ``file["version"]["metadata"]["level"]``, defaulting to
``"unknown"`` when any of those keys is absent. Levels must be strings; their
integer IDs are decoded using ``prepared_data["metadata-values"]["level"]``.

The pair ``(file_id, level)`` uniquely identifies a version.

* ``annotations-segments`` retains its existing fields and adds ``level``.
* ``annotations-versions`` lists available ``(file_id, level)`` pairs, including
  versions without speech segments or supervision regions. A recording may have only one version per level; duplicate
  levels raise ``ValueError``, including when missing metadata defaults to ``unknown``.
* ``annotations-version-regions`` contains ``file_id``, ``level``, ``start``, and
  ``end``. These are the original supervision regions, including intervals
  shorter than the training chunk duration.

For example, select word-level segments of one recording with:

.. code-block:: python

   start, end = prepared_data["audio-segments-ids"][file_id]
   segments = prepared_data["annotations-segments"][start:end]
   word = prepared_data["metadata-values"]["level"].index("word")
   word_segments = segments[segments["level"] == word]

To supervise separate versions, select available levels from
``annotations-versions`` for the recording, then filter segments and version
regions by the same ``(file_id, level)`` pair. Regions distinguish annotated silence from missing
supervision; absence of a speech segment alone does not establish silence.

Sampling and existing tasks
---------------------------

Recording-level ``annotations-regions`` contains the union of all version
regions, with the usual minimum chunk-duration filter. ``audio-annotated`` counts
this union once, so adding versions does not multiply the recording's sampling
weight. Validation chunks are likewise generated once per recording.

Existing task implementations that do not filter levels or versions continue
to consume all segments together. The cache provides the data needed for custom
version-aware targets; it does not implement head routing or a multi-version loss.

Existing caches are reused unchanged. Delete or choose a new cache path to
rebuild them with level and version fields. This also applies to earlier branch
caches that used a separate version ID.
