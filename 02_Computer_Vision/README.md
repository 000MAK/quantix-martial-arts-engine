# 02 · Computer Vision

`src/martial_arts_engine/vision.py` extracts per-person pose landmarks with MediaPipe Pose Landmarker from a user-supplied video and compatible `.task` model asset. `kinematics.py` derives projected 2-D joint angles and image-space endpoint speeds. These are geometric features, not force, physical strike speed, action labels, or injury probabilities. Add independently verified labels and group splits before model training. The optional PyTorch pose baseline is in `deep_learning.py`; it is a frame-feature MLP, not a temporal video transformer or universal technique detector.

`performance_metrics.py` adds a documented wrist-to-head guard-position proxy and separate torso/joint-to-effector cross-correlation features for annotated strike windows. These are hypothesis-generating camera-plane measurements; they do not validate actual guard quality or measure millisecond-level coordination.

Install `requirements-vision.txt`. Only process video with suitable participant consent and source rights.
