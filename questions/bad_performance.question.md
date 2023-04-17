---
title: "How can I improve performance?"
alt_titles:
  - "Pretrained pipelines do not produce good results on my data. What can I do?"
  - "It does not work! Help me!"
---

**Long answer:**

1. Manually annotate dozens of conversations as precisely as possible.
2. Separate them into train (80%), development (10%) and test (10%) subsets.
3. Setup the data for use with [`pyannote.database`](https://github.com/pyannote/pyannote-database#speaker-diarization).
4. Follow [this recipe](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/adapting_pretrained_pipeline.ipynb).
5. Enjoy.

**Also:** [I am available](https://herve.niderb.fr) for contracting to help you with that.
