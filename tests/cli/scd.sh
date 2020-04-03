#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio scd train --to=4 $GITHUB_WORKSPACE/tests/cli/scd $DEBUG
pyannote-audio scd validate --from=2 --to=4 --every=2 $GITHUB_WORKSPACE/tests/cli/scd/train/$DEBUG.train $DEBUG
pyannote-audio scd apply $GITHUB_WORKSPACE/tests/cli/scd/train/$DEBUG.train/validate_segmentation_fscore/$DEBUG.development $DEBUG
