PYANNOTE_CACHE_DUMMY=$HOME/.cache/torch/pyannote_dummy
PYANNOTE_CACHE_LOCAL=$HOME/.cache/torch/pyannote_local
# download files
PYANNOTE_CACHE=$PYANNOTE_CACHE_DUMMY python3 -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained(\"pyannote/speaker-diarization\", use_auth_token=\"ACCESS_TOKEN_GOES_HERE\")"

# copy files to dummy local directory for demonstration
cp -rL $PYANNOTE_CACHE_DUMMY/* $PYANNOTE_CACHE_LOCAL
rm -r $PYANNOTE_CACHE_DUMMY


# edit config from $PYANNOTE_CACHE_LOCAL/models--pyannote--speaker-diarization/snapshots/2c6a571d14c3794623b098a065ff95fa22da7f25/config.yaml to use local files
# embedding MUST contain speechbrain as a condition for selecting SpeechBrainPretrainedSpeakerEmbedding
# snapshot id c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b could change in the future
echo """pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: $PYANNOTE_CACHE_LOCAL/speechbrain
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: $PYANNOTE_CACHE_LOCAL/models--pyannote--segmentation/snapshots/c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b/pytorch_model.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 15
    threshold: 0.7153814381597874
  segmentation:
    min_duration_off: 0.5817029604921046
    threshold: 0.4442333667381752""" >  $PYANNOTE_CACHE_LOCAL/config.yaml

# no files should be downloaded
python3 -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained(\"$PYANNOTE_CACHE_LOCAL/config.yaml\")"
