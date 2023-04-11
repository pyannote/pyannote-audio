from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
pipeline.embedding_batch_size = 1
pipeline.to("mps")
with ProgressHook() as hook:
    diarization = pipeline("../tests/data/trn00.wav", hook=hook)
print(diarization)
