#########
Changelog
#########

Develop
~~~~~~~

- BREAKING: forbid `@` in protocol names (now used to select a version)
- BREAKING: reserve `version` key in subsets (now used to store the annotation version)
- fix: fix `X` meta-protocols with Python 3.14
- fix: `FileFinder` patterns only match files (not directories nor hidden files)
- feat: add annotation versions with optional metadata (selected with `Database.Task.Protocol@version`)
- feat: add `PYANNOTE_DATABASE_DATA_DIRS` environment variable to set root directories of relative `Databases:` paths

Version 6.1.1 (2025-12-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- fix: fix warning about invalid espace sequence

Version 6.1.0 (2025-09-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- BREAKING: update `STMLoader` to return `meeteval.io.SegLST` instances

Version 6.0.0 (2025-09-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- BREAKING: drop support to `Python` < 3.10
- BREAKING: switch to native namespace package 
- setup: switch to `uv`

Version 5.1.3 (2025-01-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~

 - chore: remove deprecated use of `delim_whitespace`
 - chore: use `importlib.metadata` instead of `pkg_resources`

Version 5.1.0 (2024-04-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING: switch to python 3.9+ (no support for 3.12 yet)
  - feat(registry): make registry iterable (yields protocol names)
  - fix(setup): fix `typer`` dependency

Version 5.0.1 (2023-04-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix support for numeric values in custom protocols (e.g. `channel: 0`)
  - improve: do not complain about missing scope for meta protocols

Version 5.0.0 (2023-03-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat(registry): add pyannote.database.registry to load multiple YAML configuration file
  - BREAKING: turn pyannote.database.get_{database|protocol|protocols} function into registry methods
  - BREAKING: remove support for database plugins (via "pyannote.database.database" entrypoint)
  - feat(protocol): add generic segmentation protocol with "classes" metadata
  - feat(protocol): add support for label "scope" in speaker diarization protocols
  - improve(protocol): make ProtocolFile pickable
  - feat(loader): add support for LAB and STM files
  - feat(loader): add option to filter RTTM lines by "type"
  - fix(cli): fix corner case when annotation key is missing

Version 4.1.3 (2022-02-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix corner case with missing uris in RTTM and UEM files
  - fix(CI): fix documentation deployment

Version 4.1.1 (2021-08-4)
~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix support for meta verification protocol

Version 4.1 (2021-04-07)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: make {RTTM|UEM}Loader usable as preprocessors
  - improve: remove the need for underscore prefix in database.yml
  - fix: fix ProtocolFile.__iter__ (dictionary changed size during iteration)

Version 4.0.4 (2020-12-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix ProtocolFile.__iter__ (dictionary changed size during iteration)

Version 4.0.3 (2020-10-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: make custom protocols pickable

Version 4.0.2 (2020-10-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: automagically crop "annotated" to file "duration" when available
  - fix: fix a bug in SpeakerDiarizationProtocols
  - improve: avoid loading ProtocolFile (lazy) keys in "get_unique_identifier"

Version 4.0.1 (2020-06-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: set "name" attribute in get_protocol
  - fix: display warning only when precomputed value is modified

Version 4.0 (2020-06-15)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for custom speaker verification protocols
  - feat: add pyannote.database.loader entrypoint
  - feat: add pyannote-database CLI
  - feat: add a few dataloaders (RTTM, UEM, CTM, MAP)
  - feat: add support for nested ProtocolFile
  - doc: major documentation update (README and docstrings)
  - BREAKING: custom protocols must define a "uri" section
  - BREAKING: remove support for "preprocessors" in Database constructor
  - BREAKING: remove support for progress bars

Version 3.0.1 (2020-03-31)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING (feat): if a "database.yml" file exists in current working directory, it will be used even if PYANNOTE_DATABASE_CONFIG is set to another value.
  - feat: add support in FileFinder for paths relative to "database.yml"
  - BREAKING: rename "config_yml" option to "database_yml" in FileFinder
  - feat: add support in custom protocols for paths relative to "database.yml" (@PaulLerner)
  - BREAKING (feat): use "annotated" to crop "annotation" in custom protocols (@PaulLerner)
  - fix: add support for int-like protocol name in custom protocols (@PaulLerner)

Version 2.5 (2020-02-04)
~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING: refactor {current | protocol}_file_iter
  - BREAKING: only rely on "uri" to decide if a ProtocolFile contains multiple files
  - BREAKING: deprecate FileFinder.current_file_iter in favor of ProtocolFile.files
  - BREAKING: deprecate FileFinder.protocol_file_iter in favor of Protocol.files
  - fix: fix support for lazy preprocessors in {Protocol | ProtocolFile}.files

Version 2.4.3 (2020-01-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix infinite recursion in "ProtocolFile" lazy evaluation

Version 2.4.2 (2020-01-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: add support for int-like database name in custom protocol

Version 2.4.1 (2019-12-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: make ProtocolFile thread-safe

Version 2.4 (2019-12-17)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: make preprocessors lazy
  - fix: pandas would convert a label to NaN ([@PaulLerner](https://github.com/PaulLerner))
  - feat: setup continuous integration
  - setup: switch to pyannote.core 3.2

Version 2.3.1 (2019-09-04)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix support for MDTM files in `pyannote.database.custom.subset_iter` ([#23](https://github.com/pyannote/pyannote-database/issues/23))

Version 2.3 (2019-07-19)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add LabelMapper preprocessor ([@MarvinLvn](https://github.com/MarvinLvn))
  - chore: replace (deprecated) pandas.read_table with pandas.read_csv ([@V-assim](https://github.com/V-assim))
  - chore: use YAML safe loader ([@V-assim](https://github.com/V-assim))

Version 2.2 (2019-06-26)
~~~~~~~~~~~~~~~~~~~~~~~~

  - setup: switch to pyannote.core 3.0
  - feat: add RTTMLoader preprocessor

Version 2.1 (2019-04-04)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for "domain" key in generic protocol

Version 2.0 (2019-03-20)
~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING: change location and format of pyannote.database configuration file
  - feat: add support for PYANNOTE_DATABASE_CONFIG environment variable

Version 1.6 (2019-03-12)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for [file-based speaker diarization protocols](https://github.com/pyannote/pyannote-database/tree/develop#generic-speaker-diarization-protocols)
  - setup: switch to pyannote.core 2.1

Version 1.5.5 (2018-11-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix Collection.files when progress=True

Version 1.5.4 (2018-11-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: skip files with no "uri" entry in FileFinder.protocol_file_iter

Version 1.5.3 (2018-11-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix broken SpeakerVerificationProtocol

Version 1.5.1 (2018-10-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix support for string preprocessors

Version 1.5 (2018-09-25)
~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING: simplify SpeakerVerificationProtocol with {subset}_trial methods

Version 1.4 (2018-07-13)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add raw collection protocol

Version 1.3.2 (2018-05-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix regression introduced in 1.3.1

Version 1.3.1 (2018-05-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix bug in `FileFinder.protocol_file_iter` with empty iterators

Version 1.3 (2018-02-04)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add `extra_keys` parameter to `{protocol | current}_file_iter`

Version 1.2.1 (2018-02-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - setup: drop support for Python 2
  - feat: add `protocol_file_iter` and `current_file_iter` to FileFinder
  - feat: add `get_label_identifier` utility function
  - fix: fix "get_unique_identifier" when "database" or "channel" is None

Version 1.1 (2017-10-13)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add speaker identification protocol
  - feat: add speaker verification protocols
  - feat: add support for list of uris in FileFinder

Version 1.0 (2017-10-02)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for "meta" protocols
  - feat: add speaker spotting protocol
  - setup: switch to pyannote.core 1.1

Version 0.12 (2017-06-28)
~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add utility functions at package root
  - doc: improve documentation
  - doc: add link to pyannote-db-template repository

Version 0.11.2 (2017-03-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix a bug with string template preprocessors
  - doc: improve documentation

Version 0.11.1 (2017-01_16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add 'get_protocol' helper function

Version 0.11 (2017-01-11)
~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for validation on training set to speaker recognition protocols
  - feat: add 'get_annotated' helper function

Version 0.10.2 (2017-01-04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix bug in FileFinder

Version 0.10.1 (2016-12-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - improve: change signature of preprocessor.__call__

Version 0.9 (2016-12-14)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add "get_unique_identifier" utility function

Version 0.8.1 (2016-12-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: fix progress bar support

Version 0.8 (2016-12-06)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add progress bar support

Version 0.7.1 (2016-12-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: add 'yield_name' parameter to speaker recognition generators

Version 0.7 (2016-12-02)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add speaker recognition protocol

Version 0.6.1 (2016-12-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add FileFinder utility class
  - fix: fix SpeakerDiarizationProtocol.stats()

Version 0.5 (2016-12-01)
~~~~~~~~~~~~~~~~~~~~~~~~

  - BREAKING: replace 'medium_template' by (more generic) 'preprocessors'

Version 0.4.1 (2016-11-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~

  - fix: rename 'speakers' to 'labels' in statistics dictionary

Version 0.4 (2016-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add a method providing global statistics about a subset

Version 0.3 (2016-09-22)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for multiple media

Version 0.2 (2016-09-21)
~~~~~~~~~~~~~~~~~~~~~~~~

  - feat: add support for 'medium_template' attribute

Version 0.1 (2016-09-20)
~~~~~~~~~~~~~~~~~~~~~~~~

  - first public version
