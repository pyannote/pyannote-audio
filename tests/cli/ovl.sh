#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio ovl train --to=4 $GITHUB_WORKSPACE/tests/cli/ovl $DEBUG
pyannote-audio ovl validate --from=2 --to=4 --every=2 $GITHUB_WORKSPACE/tests/cli/ovl/train/$DEBUG.train $DEBUG
pyannote-audio ovl apply $GITHUB_WORKSPACE/tests/cli/ovl/train/$DEBUG.train/validate_detection_fscore/$DEBUG.development $DEBUG
