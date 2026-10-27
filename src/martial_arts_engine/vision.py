from __future__ import annotations
from pathlib import Path
import csv

def extract_pose_csv(video_path:Path,model_asset:Path,out_csv:Path,every_n_frames:int=5,max_frames:int=0,video_id:str|None=None)->int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError('Pose extraction requires requirements-vision.txt (opencv-python and mediapipe).') from e
    BaseOptions=mp.tasks.BaseOptions
    VisionRunningMode=mp.tasks.vision.RunningMode
    PoseLandmarker=mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions=mp.tasks.vision.PoseLandmarkerOptions
    if not model_asset.exists(): raise FileNotFoundError('Download a compatible Pose Landmarker .task bundle and pass --model-asset.')
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {video_path}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    columns=['video_id','frame_index','timestamp_ms','person_index','image_width','image_height']+[f'pose_{i}_{axis}' for i in range(33) for axis in ('x','y','z','visibility')]
    options=PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_asset)),running_mode=VisionRunningMode.VIDEO,num_poses=2,output_segmentation_masks=False)
    rows=[]; frame_idx=0; processed=0; last_ts=-1
    try:
      with PoseLandmarker.create_from_options(options) as landmarker:
       while True:
        ok,bgr=cap.read()
        if not ok: break
        if frame_idx%max(1,every_n_frames): frame_idx+=1; continue
        pos_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        candidate=int(round(pos_ms)) if pos_ms>last_ts else int(frame_idx*1000/fps)
        ts=max(last_ts+1,candidate); last_ts=ts
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        result=landmarker.detect_for_video(image,ts)
        for person_idx,landmarks in enumerate(result.pose_landmarks):
         rec={'video_id':video_id or video_path.stem,'frame_index':frame_idx,'timestamp_ms':ts,'person_index':person_idx,'image_width':int(bgr.shape[1]),'image_height':int(bgr.shape[0])}
         for i,lm in enumerate(landmarks):
          rec[f'pose_{i}_x']=lm.x;rec[f'pose_{i}_y']=lm.y;rec[f'pose_{i}_z']=lm.z;rec[f'pose_{i}_visibility']=getattr(lm,'visibility',0.0)
         rows.append(rec)
        processed+=1
        frame_idx+=1
        if max_frames and processed>=max_frames: break
    finally: cap.release()
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    return len(rows)
