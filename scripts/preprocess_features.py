# scripts/preprocess_features.py
# FINAL CORRECTED VERSION
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import yaml
from tqdm import tqdm
from dotmap import DotMap
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from einops import rearrange

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_video_frames(video_path, num_frames, resize_shape=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2: return None
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = TF.to_tensor(cv2.resize(frame, resize_shape))
            frames.append(frame_tensor)
        else:
            cap.release(); return None
    cap.release()
    return torch.stack(frames) if len(frames) == num_frames else None

def compute_optical_flow(frames, raft_model, device, config):
    # --- THIS IS THE FINAL, CORRECTED LOGIC ---
    T_frames, C, H, W = frames.shape
    patch_size = config.model.video_patch_size

    frames = frames.to(device) * 255.0
    frame_pairs_1, frame_pairs_2 = frames[:-1], frames[1:]
    
    with torch.no_grad():
        flow_preds = raft_model(frame_pairs_1, frame_pairs_2)[-1]
    
    avg_flow_map = torch.mean(flow_preds, dim=0) # Shape: [2, H, W]

    # Correctly average the flow within each patch grid.
    # 1. Rearrange into patches: [2, H, W] -> [196, 2, 16, 16]
    flow_in_patches = rearrange(avg_flow_map, 'c (ph p1) (pw p2) -> (ph pw) c p1 p2', p1=patch_size, p2=patch_size)
    # 2. Average over the patch dimensions (16x16): -> [196, 2]
    avg_flow_per_patch = torch.mean(flow_in_patches, dim=(-1, -2))
    
    flow_dim = config.model.moe.experts.motion.flow_dim
    # 3. Pad the feature dimension (2) to the desired final dimension (64)
    return F.pad(avg_flow_per_patch, (0, flow_dim - 2)).cpu() # [196, 2] -> [196, 64]

def compute_frame_deltas(frames, config):
    num_patches_per_frame = (224 // config.model.video_patch_size) ** 2
    deltas = (frames[1:] - frames[:-1]).abs().mean()
    return torch.full((num_patches_per_frame, config.model.moe.experts.fast_change.delta_dim), deltas.item()).cpu()

if __name__ == '__main__':
    with open("config/training_msvd.yaml") as f: config = DotMap(yaml.safe_load(f))
    
    DATA_ROOT, VIDEO_DIR = config.data.data_root, os.path.join(config.data.data_root, "videos")
    FLOW_DIR, DELTAS_DIR = os.path.join(DATA_ROOT, "flow"), os.path.join(DATA_ROOT, "deltas")
    os.makedirs(FLOW_DIR, exist_ok=True); os.makedirs(DELTAS_DIR, exist_ok=True)
    
    device = get_device(); print(f"Using device: {device}")
    weights = Raft_Small_Weights.DEFAULT; raft_model = raft_small(weights=weights).to(device); raft_model.eval()
    
    video_filenames = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    print(f"Found {len(video_filenames)} AVI files to process.")

    for video_filename in tqdm(video_filenames, desc="Processing videos"):
        video_id_no_ext = os.path.splitext(video_filename)[0]
        flow_path = os.path.join(FLOW_DIR, f"{video_id_no_ext}.pt")
        delta_path = os.path.join(DELTAS_DIR, f"{video_id_no_ext}.pt")

        video_path = os.path.join(VIDEO_DIR, video_filename)
        frames = load_video_frames(video_path, config.model.frames_per_video)
        if frames is None:
            print(f"Warning: Could not load frames for {video_filename}. Skipping.")
            continue
            
        flow_vectors = compute_optical_flow(frames, raft_model, device, config)
        delta_vectors = compute_frame_deltas(frames, config)
        torch.save(flow_vectors, flow_path)
        torch.save(delta_vectors, delta_path)
            
    print("\nPre-computation of REAL feature files complete.")