# pyannote-database

Reproducible experimental protocols for multimedia (audio, video, text) database.

```bash
$ pip install pyannote.database
```

- [pyannote-database](#pyannote-database)
  - [Definitions](#definitions)
  - [Configuration file](#configuration-file)
  - [Data loaders](#data-loaders)
  - [Preprocessors](#preprocessors)
  - [`FileFinder`](#filefinder)
  - [Tasks](#tasks)
    - [Collections](#collections)
    - [Segmentation](#segmentation)
    - [Speaker diarization](#speaker-diarization)
    - [Speaker verification](#speaker-verification)
  - [Versioning](#versioning)
  - [Meta-protocols and requirements](#meta-protocols-and-requirements)
  - [Plugins](#plugins)
  - [API](#api)
    - [Databases and tasks](#databases-and-tasks)
    - [Custom data loaders](#custom-data-loaders)
      - [Defining custom data loaders](#defining-custom-data-loaders)
      - [Registering custom data loaders](#registering-custom-data-loaders)
      - [Testing custom data loaders](#testing-custom-data-loaders)
    - [Protocols](#protocols)
      - [Collections](#collections-1)
      - [Speaker diarization](#speaker-diarization-1)
      - [Speaker verification](#speaker-verification-1)

## Definitions

In `pyannote.database` jargon, a **resource** can be any multimedia entity (e.g. an image, an audio file, a video file, or a webpage). In its most simple  form, it is modeled as a `pyannote.database.ProtocolFile` instance (basically a `dict` on steroids) with a `uri` key (URI stands for _unique resource identifier_) that identifies the entity.

**Metadata** may be associated to a resource by adding keys to its `ProtocolFile`. For instance, one could add a `label` key to an image _resource_ describing whether it depicts a chihuahua or a muffin.

A **database** is a collection of resources of the same nature (e.g. a collection of audio files). It is modeled as a `pyannote.database.Database` instance.

An **experimental protocol** (`pyannote.database.Protocol`) usually defines three subsets:

- a _train_ subset (_e.g._ used to train a neural network),
- a _development_ subset (_e.g._ used to tune hyper-parameters),
- a _test_ subset (_e.g._ used for evaluation).

## Configuration file

Experimental protocols are defined via YAML configuration files:

  ```yaml
  Protocols:
    MyDatabase:
      Protocol:
        MyProtocol:
          train:
              uri: /path/to/train.lst
          development:
              uri: /path/to/development.lst
          test:
              uri: /path/to/test.lst
  ```

where `/path/to/train.lst` contains the list of unique resource identifier (URI) of the
files in the _train_ subset:

  ```text
  # /path/to/train.lst
  filename1
  filename2
  ```

Since version `5.0`, configuration files must be loaded into the `registry` like that: 

```python
from pyannote.database import registry
registry.load_database("/path/to/database.yml")
```

`registry.load_database` takes an optional `mode` keyword argument that controls what to do when loading a protocol whose name (e.g. `MyDatabase.Protocol.MyProtocol`) is already used by another protocol:
 - `LoadingMode.OVERRIDE` to override existing protocol by the new one (default behavior);
 - `LoadingMode.KEEP` to keep existing protocol;
 - `LoadingMode.ERROR` to raise a `RuntimeException` when such a conflict occurs.

For backward compatibility with `4.x` branch, the following configuration files are loaded automatically when importing `pyannote.database`, in that order:

  1. `~/.pyannote/database.yml`
  2. `database.yml` in current working directory
  3. list of `;`-separated path(s) in the `PYANNOTE_DATABASE_CONFIG` environment variable (e.g. `/absolute/path.yml;relative/path.yml`)

Once loaded in the registry, protocols can be used in Python like this:

  ```python
  from pyannote.database import registry
  registry.load_database("/path/to/database.yml")

  protocol = registry.get_protocol('MyDatabase.Protocol.MyProtocol')
  for resource in protocol.train():
      print(resource["uri"])
  filename1
  filename2
  ```

Paths defined in the configuration file can be absolute or relative to the directory containing the configuration file. 
For instance, the following file organization should work just fine:

  ```text
  .
  ├── database.yml
  └── lists
      └── train.lst
  ```

with the content of `database.yml` as follows:

  ```yaml
  Protocols:
    MyDatabase:
      Protocol:
        MyProtocol:
          train:
              uri: lists/train.lst
  ```

## Data loaders

The above `MyDatabase.Protocol.MyProtocol` protocol is not very useful as it only allows to iterate over a list of resources with a single `'uri'` key. Metadata can be added to each resource with the following syntax:  

  ```yaml
  Protocols:
    MyDatabase:
      Protocol:
        MyProtocol:
          train:
              uri: lists/train.lst
              speaker: rttms/train.rttm
              transcription: ctms/{uri}.ctm
  ```

and the following directory structure:

  ```text
  .
  ├── database.yml
  ├── lists
  |   └── train.lst
  ├── rttms
  |   └── train.rttm
  └── ctms
      ├── filename1.ctm
      └── filename2.ctm
  ```

Now, resources have both `'speaker'` and `'transcription'` keys:

  ```python
  from pyannote.database import registry
  protocol = registry.get_protocol('MyDatabase.Protocol.MyProtocol')

  for resource in protocol.train():
      assert "speaker" in resource
      assert isinstance(resource["speaker"], pyannote.core.Annotation)
      assert "transcription" in resource
      assert isinstance(resource["transcription"], spacy.tokens.Doc)
  ```

What happened exactly? Data loaders were automatically selected based on metadata file suffix:

- `pyannote.database.loader.RTTMLoader` for `speaker` entry with `.rttm` suffix
- `pyannote.database.loader.CTMLoader` for  `transcription` entry with `ctm` suffix).

and used to populate `speaker` and `transcription` keys. In pseudo-code:

  ```python
  # instantiate loader registered with `.rttm` suffix
  speaker = RTTMLoader('rttms/train.rttm')

  # entries with {placeholders} serve as path templates
  transcription_template = 'ctms/{uri}.ctm'

  for resource in protocol.train():
      # unique resource identifier
      uri = resource['uri']

      # only select parts of `rttms/train.rttm` that are relevant to current resource,
      # convert it into a convenient data structure (here pyannote.core.Annotation), 
      # and assign it to `'speaker'` resource key 
      resource['speaker'] = speaker[uri]

      # replace placeholders in `transcription` path template
      ctm = transcription_template.format(uri=uri)

      # instantiate loader registered with `.ctm` suffix
      transcription = CTMLoader(ctm)

      # only select parts of the `ctms/{uri}.ctm` that are relevant to current resource
      # (here, most likely the whole file), convert it into a convenient data structure
      # (here spacy.tokens.Doc), and assign it to `'transcription'` resource key 
      resource['transcription'] = transcription[uri]
  ```

`pyannote.database` provides built-in data loaders for a limited set of file formats: `RTTMLoader` for `.rttm` files, `UEMLoader` for `.uem` files, and `CTMLoader` for `.ctm` files. See [Custom data loaders](#custom-data-loaders) section to learn how to add your own. 

## Preprocessors

When iterating over a protocol subset (e.g. using `for resource in protocol.train()`), resources are provided as instances of `pyannote.database.ProtocolFile`, which are basically `dict` instances whose values are computed lazily.

For instance, in the code above, the value returned by `resource['speaker']` is only computed the first time it is accessed and then cached for all subsequent calls.  See [Custom data loaders](#custom-data-loaders) section for more details. 

Similarly, resources can be augmented (or modified) on-the-fly with the `preprocessors` options for `get_protocol`. In the example below, a `dummy` key is added that simply returns the length of the `uri` string:

```python

def compute_dummy(resource: ProtocolFile):
    print(f"Computing 'dummy' key")
    return len(resource["uri"])

from pyannote.database import registry
protocol = registry.get_protocol('Etape.SpeakerDiarization.TV', 
                                 preprocessors={"dummy": compute_dummy})
resource = next(protocol.train())
resource["dummy"]
Computing 'dummy' key
```

## `FileFinder`

`FileFinder` is a special case of preprocessors is `pyannote.database.FileFinder` meant to automatically locate the media file associated with the `uri`. 

Say audio files are available at the following paths:

  ```text
  .
  └── /path/to
      └── audio
          ├── filename1.wav
          ├── filename2.mp3
          ├── filename3.wav
          ├── filename4.wav
          └── filename5.mp3
  ```

The `FileFinder` preprocessor relies on a `Databases:` section that should be added to the `database.yml` configuration files and indicates where to look for media files (using resource key placeholders):

  ```yaml
  Databases:
    MyDatabase: 
      - /path/to/audio/{uri}.wav
      - /path/to/audio/{uri}.mp3

Protocols:
    MyDatabase:
      Protocol:
        MyProtocol:
          train:
              uri: lists/train.lst
  ```

Note that `*` and `**` patterns are supported. Paths can also be relative to the location of `database.yml`. It will then do its best to locate the file at runtime:

  ```python
  from pyannote.database import registry
  from pyannote.database import FileFinder
  protocol = registry.get_protocol('MyDatabase.SpeakerDiarization.MyProtocol', 
                                   preprocessors={"audio": FileFinder()})
  for resource in protocol.train():
      print(resource["audio"])
  /path/to/audio/filename1.wav
  /path/to/audio/filename2.mp3
  ```

Relative paths can also be looked for in other directories, listed in the `PYANNOTE_DATABASE_DATA_DIRS` environment variable (separated by `:`, or `;` on Windows):

  ```bash
  export PYANNOTE_DATABASE_DATA_DIRS=/mnt/data:/home/me/data
  ```

  ```yaml
  Databases:
    MyDatabase: audio/{uri}.wav
  ```

`FileFinder` then looks for `/mnt/data/audio/{uri}.wav` and `/home/me/data/audio/{uri}.wav`, and only if neither exists, for `audio/{uri}.wav` next to `database.yml`. A file found in more than one directory raises an error. Absolute paths are not affected.

## Tasks

### Collections

A raw collection of files (i.e. without any train/development/test split) can be defined using the `Collection` task:

  ```yaml
  # ~/database.yml
  Protocols:
    MyDatabase:
      Collection:
        MyCollection:
          uri: /path/to/collection.lst
          any_other_key: ... # see custom loader documentation
  ```

where `/path/to/collection.lst` contains the list of identifiers of the
files in the collection:

  ```text
  # /path/to/collection.lst
  filename1
  filename2
  filename3
  ```

It can the be used in Python like this:

  ```python
  from pyannote.database import registry
  collection = registry.get_protocol('MyDatabase.Collection.MyCollection')

  for file in collection.files():
     print(file["uri"])
  filename1
  filename2
  filename3
  ```   

### Segmentation

A (temporal) segmentation protocol can be defined using the `Segmentation` task:

  ```yaml
  Protocols:
    MyDatabase:
      Segmentation:
        MyProtocol:
          classes: 
            - speech
            - noise
            - music
          train:
              uri: /path/to/train.lst
              annotation: /path/to/train.rttm
              annotated: /path/to/train.uem
  ```

where `/path/to/train.lst` contains the list of identifiers of the
files in the training set:

  ```text
  # /path/to/train.lst
  filename1
  filename2
  ```

`/path/to/train.rttm` contains the reference segmentation using
RTTM format:

  ```text
  # /path/to/reference.rttm
  SPEAKER filename1 1 3.168 0.800 <NA> <NA> speech <NA> <NA>
  SPEAKER filename1 1 5.463 0.640 <NA> <NA> speech <NA> <NA>
  SPEAKER filename1 1 5.496 0.574 <NA> <NA> music <NA> <NA>
  SPEAKER filename1 1 10.454 0.499 <NA> <NA> music <NA> <NA>
  SPEAKER filename2 1 2.977 0.391 <NA> <NA> noise <NA> <NA>
  SPEAKER filename2 1 18.705 0.964 <NA> <NA> noise <NA> <NA>
  SPEAKER filename2 1 22.269 0.457 <NA> <NA> speech <NA> <NA>
  SPEAKER filename2 1 28.474 1.526 <NA> <NA> speech <NA> <NA>
  ```

`/path/to/train.uem` describes the annotated regions using UEM format:

  ```text
  filename1 NA 0.000 30.000
  filename2 NA 0.000 30.000
  filename2 NA 40.000 70.000
  ```

It is recommended to provide the `annotated` key even if it covers the whole file. Any part of `annotation` that lives outside of the provided `annotated` will  be removed. It is also used by `pyannote.metrics` to remove un-annotated regions from the evaluation, and to prevent `pyannote.audio` from incorrectly considering empty un-annotated regions as negatives.

It can then be used in Python like this:

  ```python
  from pyannote.database import registry
  protocol = registry.get_protocol('MyDatabase.Segmentation.MyProtocol')

  for file in protocol.train():
     print(file["uri"])
     assert "annotation" in file
     assert "annotated" in file
  filename1
  filename2
  ```

### Speaker diarization

A protocol can be defined specifically for speaker diarization using the `SpeakerDiarization` task:

  ```yaml
  Protocols:
    MyDatabase:
      SpeakerDiarization:
        MyProtocol:
          scope: file
          train:
              uri: /path/to/train.lst
              annotation: /path/to/train.rttm
              annotated: /path/to/train.uem
  ```

where `/path/to/train.lst` contains the list of identifiers of the
files in the training set:

  ```text
  # /path/to/train.lst
  filename1
  filename2
  ```

`/path/to/train.rttm` contains the reference speaker diarization using
RTTM format:

  ```text
  # /path/to/reference.rttm
  SPEAKER filename1 1 3.168 0.800 <NA> <NA> speaker_A <NA> <NA>
  SPEAKER filename1 1 5.463 0.640 <NA> <NA> speaker_A <NA> <NA>
  SPEAKER filename1 1 5.496 0.574 <NA> <NA> speaker_B <NA> <NA>
  SPEAKER filename1 1 10.454 0.499 <NA> <NA> speaker_B <NA> <NA>
  SPEAKER filename2 1 2.977 0.391 <NA> <NA> speaker_C <NA> <NA>
  SPEAKER filename2 1 18.705 0.964 <NA> <NA> speaker_C <NA> <NA>
  SPEAKER filename2 1 22.269 0.457 <NA> <NA> speaker_A <NA> <NA>
  SPEAKER filename2 1 28.474 1.526 <NA> <NA> speaker_A <NA> <NA>
  ```

`/path/to/train.uem` describes the annotated regions using UEM format:

  ```text
  filename1 NA 0.000 30.000
  filename2 NA 0.000 30.000
  filename2 NA 40.000 70.000
  ```

It is recommended to provide the `annotated` key even if it covers the whole file. Any part of `annotation` that lives outside of the provided `annotated` will  be removed. It is also used by `pyannote.metrics` to remove un-annotated regions from the evaluation, and to prevent `pyannote.audio` from incorrectly considering empty un-annotated regions as non-speech.

It can then be used in Python like this:

  ```python
  from pyannote.database import registry
  protocol = registry.get_protocol('MyDatabase.SpeakerDiarization.MyProtocol')

  for file in protocol.train():
     print(file["uri"])
     assert "annotation" in file
     assert "annotated" in file
  filename1
  filename2
  ```

The `scope` parameters indicates the scope of speaker labels:
  * `file` indicates that each file has its own set of speaker labels. There is no guarantee that `speaker1` in `filename1` is the same speaker as `speaker1` in `filename2`. 
  * `database` indicates that all files in the database share the same set of speaker labels. `speaker1` in `database1/filename1` is the same speaker as `speaker1` in `database1/filename2`.
  * `global` indicates that the set of speaker labels is the same across all databases. `speaker1` in `database1` is the same speaker as `speaker1` in `database2`. 

`scope` is then directly accessible from `file['scope']`.

### Speaker verification

A simple speaker verification protocol can be defined by adding a `trial` entry to a `SpeakerVerification` task:

  ```yaml
  Protocols:
    MyDatabase:
      SpeakerVerification:
        MyProtocol:
          train:
              uri: /path/to/train.lst
              duration: /path/to/duration.map
              trial: /path/to/trial.txt
  ```

where `/path/to/train.lst` contains the list of identifiers of the
files in the collection:

  ```text
  # /path/to/collection.lst
  filename1
  filename2
  filename3
  ...
  ```

`/path/to/duration.map` contains the duration of the files:

  ```text
  filename1 30.000
  filename2 30.000
  ...
  ```

`/path/to/trial.txt` contains a list of trials :

  ```text
  1 filename1 filename2
  0 filename1 filename3
  ...
  ```

`1` stands for _target_ trials and `0` for _non-target_ trials.
In the example below, it means that the same speaker uttered files `filename1` and `filename2` and that `filename1` and `filename3` are from two different speakers. 

It can then be used in Python like this:

  ```python
  from pyannote.database import registry
  protocol = registry.get_protocol('MyDatabase.SpeakerVerification.MyProtocol')

  for trial in protocol.train_trial():
     print(f"{trial['reference']} {trial['file1']['uri']} {trial['file2']['uri']}")
  1 filename1 filename2
  0 filename1 filename3
  ```

Note that speaker verification protocols (`SpeakerVerificationProtocol`) are a subclass of speaker diarization protocols (`SpeakerDiarizationProtocol`). As such, they also define regular `{subset}` methods.

## Versioning

A protocol can provide several versions of its annotations (e.g. original and forced-aligned) using `versions:` and the `{version}` placeholder:

```yaml
Protocols:
  MyDatabase:
    SpeakerDiarization:
      MyProtocol:
        versions:
          original:
          forced_alignment:
            closing: 0.2
            aligner: mfa
        scope: file
        train:
          uri: /path/to/train.lst
          annotation: /path/to/{version}/train.rttm
          annotated: /path/to/train.uem
```

where `{version}` is replaced by the version name. Each version can have optional metadata (anything describing it), passed as is to every file.

Select a version by appending `@version` to the protocol name:

```python
from pyannote.database import registry
protocol = registry.get_protocol('MyDatabase.SpeakerDiarization.MyProtocol@forced_alignment')

for file in protocol.train():
    file["annotation"]  # loaded from /path/to/forced_alignment/train.rttm
    file["version"]     # {"name": "forced_alignment", "metadata": {"closing": 0.2, "aligner": "mfa"}}
```

Without `@version`, the first version listed (here `original`) is used.

Note that protocol names cannot contain `@`, version names cannot contain `.`, and `version` cannot be used as a subset key.

## Meta-protocols and requirements

`pyannote.database` provides a way to combine several protocols (possibly
from different databases) into one.

This is achieved by defining those "meta-protocols" into the configuration file with the special `X` database:

  ```yaml
  Requirements:
    - /path/to/my/database/database.yml         # defines MyDatabase protocols
    - /path/to/my/other/database/database.yml   # defines MyOtherDatabase protocols

  Protocols:
    X:
      Protocol:
        MyMetaProtocol:
          train:
            MyDatabase.Protocol.MyProtocol: [train, development]
            MyOtherDatabase.Protocol.MyOtherProtocol: [train, ]
          development:
            MyDatabase.Protocol.MyProtocol: [test, ]
            MyOtherDatabase.Protocol.MyOtherProtocol: [development, ]
          test:
            MyOtherDatabase.Protocol.MyOtherProtocol: [test, ]
```

The new `X.Protocol.MyMetaProtocol` combines the `train` and `development` subsets of `MyDatabase.Protocol.MyProtocol` with the `train` subset of `MyOtherDatabase.Protocol.MyOtherProtocol` to build a meta `train` subset.

This new "meta-protocol" can be used like any other protocol of the (fake) `X` database:

  ```python
  from pyannote.database import registry
  protocol = registry.get_protocol('X.Protocol.MyMetaProtocol')
  
  for resource in protocol.train():
      pass
  ```

## Plugins

For more complex protocols, you can create (and share) your own [`pyannote.database` plugin](http://github.com/pyannote/pyannote-db-template).

A bunch of `pyannote.database` plugins are already available (search for `pyannote.db` on [pypi](https://pypi.python.org/pypi?%3Aaction=search&term=pyannote.db&submit=search))

## API

### Databases and tasks

Everything about databases is stored in the `registry`.

```python
  from pyannote.database import registry
```

Any database can then be instantiated as follows:

  ```python
  database = registry.get_database("MyDatabase")
  ```

Some databases (especially multimodal ones) may be used for several tasks.
One can get a list of tasks using `get_tasks` method:

  ```python
  database.get_tasks()
  ["SpeakerDiarization"]
  ```


### Custom data loaders

`pyannote.database` provides built-in data loaders for a limited set of file formats: `RTTMLoader` for `.rttm` files, `UEMLoader` for `.uem` files, and `CTMLoader` for `.ctm` files.

In case those are not enough, `pyannote.database` supports the addition of custom data loaders using the `pyannote.database.loader` entry point.

#### Defining custom data loaders

Here is an example of a Python package called `your_package` that defines two custom data loaders for files with `.ext1` and `.ext2` suffix respectively.

```python
# ~~~~~~~~~~~~~~~~ YourPackage/your_package/loader.py ~~~~~~~~~~~~~~~~
from pyannote.database import ProtocolFile
from pathlib import Path

class Ext1Loader:
    def __init__(self, ext1: Path):
        print(f'Initializing Ext1Loader with {ext1}')
        # your code should obviously do something smarter.
        # see pyannote.database.loader.RTTMLoader for an example.
        self.ext1 = ext1

    def __call__(self, current_file: ProtocolFile) -> Text:
        uri = current_file["uri"]
        print(f'Processing {uri} with Ext1Loader')
        # your code should obviously do something smarter.
        # see pyannote.database.loader.RTTMLoader for an example.
        return f'{uri}.ext1'

class Ext2Loader:
    def __init__(self, ext2: Path):
        print(f'Initializing Ext2Loader with {ext2}')
        # your code should obviously do something smarter.
        # see pyannote.database.loader.RTTMLoader for an example.
        self.ext2 = ext2

    def __call__(self, current_file: ProtocolFile) -> Text:
        uri = current_file["uri"]
        print(f'Processing {uri} with Ext2Loader')
        # your code should obviously do something smarter.
        # see pyannote.database.loader.RTTMLoader for an example.
        return f'{uri}.ext2'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

The `__init__` method expects a unique positional argument of type `Path` that provides the path to the data file in the custom data format.

`__call__` expects a unique positional argument of type `ProtocolFile` and returns the data for the given file. 

It is recommended to make `__init__` as fast and light as possible and delegate all the data filtering and formatting to `__call__`. For instance, `RTTMLoader.__init__` uses `pandas` to load the full `.rttm` file as fast as possible in a `DataFrame`, while `RTTMLoader.__call__` takes care of selecting rows that correspond to the requested file and convert them into a `pyannote.core.Annotation`. 

#### Registering custom data loaders

At this point, `pyannote.database` has no idea of the existence of these new custom data loaders. They must be registered using the `pyannote.database.loader` entry-point in `your_package`'s `setup.py`, and then install the library `pip install your_package` (or `pip install -e YourPackage/` if it is not published on PyPI yet).

```python
# ~~~~~~~~~~~~~~~~~~~~~~~ YourPackage/setup.py ~~~~~~~~~~~~~~~~~~~~~~~
from setuptools import setup, find_packages
setup(
    name="your_package",
    packages=find_packages(),
    install_requires=[
        "pyannote.database >= 4.0",
    ]
    entry_points={
        "pyannote.database.loader": [
            # load files with extension '.ext1' 
            # with your_package.loader.Ext1Loader
            ".ext1 = your_package.loader:Ext1Loader",
            # load files with extension '.ext2' 
            # with your_package.loader.Ext2Loader
            ".ext2 = your_package.loader:Ext2Loader",
        ],
    }
)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

#### Testing custom data loaders

Now that `.ext1` and `.ext2` data loaders are registered, they will be used automatically by `pyannote.database` when parsing the sample `demo/database.yml` custom protocol configuration file.

```yaml
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ demo/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~
Protocols:
  MyDatabase:
    SpeakerDiarization:
      MyProtocol:
        train:
           uri: train.lst
           key1: train.ext1
           key2: train.ext2
```

```python
# tell pyannote.database about the configuration file
>>> from pyannote.database import registry
>>> registry.load_database('demo/database.yml')

# load custom protocol
>>> protocol = registry.get_protocol('MyDatabase.SpeakerDiarization.MyProtocol')

# get first file of training set
>>> first_file = next(protocol.train())
Initializing Ext1Loader with file train.ext1
Initializing Ext2Loader with file train.ext2

# access its "key1" and "key2" keys.
>>> assert first_file["key1"] == 'fileA.ext1'
Processing fileA with Ext1Loader
>>> assert first_file["key2"] == 'fileA.ext2'
Processing fileA with Ext2Loader
# note how __call__ is only called now (and not before)
# this is why it is better to delegate all the filtering and formatting to __call__

>>> assert first_file["key1"] == 'fileA.ext1'
# note how __call__ is not called the second time thanks to ProtocolFile built-in cache
```

### Protocols

An experimental protocol can be defined programmatically by creating a 
class that inherits from SpeakerDiarizationProtocol and implements at least
one of `train_iter`, `development_iter` and `test_iter` methods:

  ```python
  class MyProtocol(Protocol):
      def train_iter(self) -> Iterator[Dict]:
          yield {"uri": "filename1", "any_other_key": "..."}
          yield {"uri": "filename2", "any_other_key": "..."}
  ```

`{subset}_iter` should return an iterator of dictionnaries with 
    - "uri" key (mandatory) that provides a unique file identifier (usually
      the filename),
    - any other key that the protocol may provide.

It can then be used in Python like this:

  ```python
  protocol = MyProtocol()
  for file in protocol.train():
      print(file["uri"])
  filename1
  filename2
  ```

#### Collections

A collection can be defined programmatically by creating a class that 
inherits from CollectionProtocol and implements the `files_iter` method:

  ```python
  class MyCollection(CollectionProtocol):
      def files_iter(self) -> Iterator[Dict]:
          yield {"uri": "filename1", "any_other_key": "..."}
          yield {"uri": "filename2", "any_other_key": "..."}
          yield {"uri": "filename3", "any_other_key": "..."}
  ```

`files_iter` should return an iterator of dictionnaries with 
    - a mandatory "uri" key that provides a unique file identifier (usually
      the filename),
    - any other key that the collection may provide.

It can then be used in Python like this:

  ```python
  collection = MyCollection()
  for file in collection.files():
     print(file["uri"])
  filename1
  filename2
  filename3
  ```

#### Speaker diarization

A speaker diarization protocol can be defined programmatically by creating
a class that inherits from SpeakerDiarizationProtocol and implements at 
least one of `train_iter`, `development_iter` and `test_iter` methods:

  ```python
  class MySpeakerDiarizationProtocol(SpeakerDiarizationProtocol):
      def train_iter(self) -> Iterator[Dict]:
          yield {"uri": "filename1",
                 "annotation": Annotation(...),
                 "annotated": Timeline(...)}
          yield {"uri": "filename2",
                 "annotation": Annotation(...),
                 "annotated": Timeline(...)}
  ```

`{subset}_iter` should return an iterator of dictionnaries with

- "uri" key (mandatory) that provides a unique file identifier (usually
  the filename),
- "annotation" key (mandatory for train and development subsets) that 
  provides reference speaker diarization as a `pyannote.core.Annotation`
  instance,
- "annotated" key (recommended) that describes which part of the file 
  has been annotated, as a `pyannote.core.Timeline` instance. Any part
  of "annotation" that lives outside of the provided "annotated" will 
  be removed. This is also used by `pyannote.metrics` to remove 
  un-annotated regions from its evaluation report, and by 
  `pyannote.audio` to not consider empty un-annotated regions as 
  non-speech. 
- any other key that the protocol may provide.

It can then be used in Python like this:

  ```python
  protocol = MySpeakerDiarizationProtocol()
  for file in protocol.train():
     print(file["uri"])
  filename1
  filename2
  ```

#### Speaker verification

A speaker verification protocol implement the `{subset}_trial` functions, useful in speaker verification validation process. Note that `SpeakerVerificationProtocol` is a subclass of [SpeakerDiarizationProtocol](#speaker-diarization-1). As such, it shares the same `{subset}_iter` methods, and need a mandatory `{subset}_iter` method.

A speaker verification protocol can be defined programmatically by creating a class that inherits from `SpeakerVerificationProtocol` and implement at least one of `train_trial_iter`, `development_trial_iter` and `test_trial_iter` methods: 

  ```python
  class MySpeakerVerificationProtocol(SpeakerVerificationProtocol):
      def train_iter(self) -> Iterator[Dict]:
          yield {"uri": "filename1",
                 "annotation": Annotation(...),
                 "annotated": Timeline(...)}
          yield {"uri": "filename2",
                 "annotation": Annotation(...),
                 "annotated": Timeline(...)}
      def train_trial_iter(self) -> Iterator[Dict]:
          yield {"reference": 1,
                 "file1": ProtocolFile(...),
                 "file2": ProtocolFile(...)}
          yield {"reference": 0,
                 "file1": {
                   "uri":"filename1",
                   "try_with":Timeline(...)
                    },
                 "file1": {
                   "uri":"filename3",
                   "try_with":Timeline(...)
                   }
                 }
  ```

`{subset}_trial_iter` should return an iterator of dictionnaries with

- `reference` key (mandatory) that provides an int portraying whether `file1` and `file2` are uttered by the same speaker (1 is same, 0 is different),
- `file1` key (mandatory) that provides the first file,
- `file2` key (mandatory) that provides the second file.

Both `file1` and `file2` should be provided as dictionaries or `pyannote.database.protocol.protocol.ProtocolFile` instances with 

- `uri` key (mandatory),
- `try_with` key (mandatory) that describes which part of the file should be used in the validation process, as a `pyannote.core.Timeline` instance.
- any other key that the protocol may provide.

It can then be used in Python like this:

  ```python
  protocol = MySpeakerVerificationProtocol()
  for trial in protocol.train_trial():
     print(f"{trial['reference']} {trial['file1']['uri']} {trial['file2']['uri']}")
  1 filename1 filename2
  0 filename1 filename3
  ```
