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
---------------------

The level is read from ``file["version"]["metadata"]["level"]``, defaulting to
``"mixed"`` when any of those keys is absent. Levels must be strings; their
integer IDs are decoded using ``prepared_data["metadata-values"]["level"]``.

* ``annotations-segments`` retains its existing fields and adds ``level`` and
  ``version_id``.
* ``annotations-versions`` contains ``file_id`` and ``level``. Its row index
  is the version ID. Every protocol entry creates a version, including
  silent versions and distinct versions with the same level.
* ``annotations-version-regions`` contains ``version_id``, ``start``, and
  ``end``. These are the original supervision regions, including intervals
  shorter than the training chunk duration.

For example, select word-level segments of one recording with:

.. code-block:: python

   start, end = prepared_data["audio-segments-ids"][file_id]
   segments = prepared_data["annotations-segments"][start:end]
   word = prepared_data["metadata-values"]["level"].index("word")
   word_segments = segments[segments["level"] == word]

To supervise separate versions, select rows from ``annotations-versions``
by ``file_id`` and ``level``, then filter both segments and version regions by
their ``version_id``. Regions distinguish annotated silence from missing
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
rebuild them with level and version fields.
