# 03 · Acoustic Engine

`src/martial_arts_engine/audio.py` extracts MFCC and acoustic summary features from files named in a licensed manifest or extracted from a video when `ffmpeg` is available. These measurements are not validated strike-impact labels or coach-command transcription. Any classifier requires labels, session/athlete-grouped splits and separate evaluation. Install `requirements-audio.txt`.
