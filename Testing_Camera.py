"""
Real-time hand pose estimation from webcam
"""
import os
import cv2
import torch
import numpy as np

from Training import UNet2DHandPose
from utils import softmax_coords, weighted_local_peak_coords, max_peak_coords


# Hand skeleton structure
SKELETON_CHAINS = [
    [0, 1, 2, 3, 4],      # Thumb
    [0, 5, 6, 7, 8],      # Index
    [0, 9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20],  # Pinky
]

# Smoothing parameter
SMOOTHING_ALPHA = 0.5


def draw_skeleton(frame, coords, color):
    """
    Draw hand skeleton on frame
    
    Args:
        frame: Image frame
        coords: Coordinates [2, 21] (row, col)
        color: RGB color tuple
    """
    # Draw lines
    for chain in SKELETON_CHAINS:
        points = np.array([
            (int(coords[1, j]), int(coords[0, j]))
            for j in chain
        ])
        cv2.polylines(frame, [points], False, color, 4)
    
    # Draw joints
    for j in range(coords.shape[1]):
        x, y = int(coords[1, j]), int(coords[0, j])
        cv2.circle(frame, (x, y), 6, color, -1)


def main():
    """Main webcam inference loop"""
    print("=" * 70)
    print(" " * 15 + "Real-Time Hand Pose Estimation")
    print("=" * 70)
    
    # Select model
    models_dir = "models/"
    models = os.listdir(models_dir)
    
    print("\nAvailable models:")
    for i, model_name in enumerate(models):
        print(f"  {i}. {model_name}")
    
    model_idx = int(input("\nEnter model index: "))
    model_path = os.path.join(models_dir, models[model_idx])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DHandPose().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\nLoaded model: {models[model_idx]}")
    print(f"Device: {device}")
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    h, w, _ = frame.shape
    print(f"Camera resolution: {w}x{h}")
    
    # Video writer (optional)
    save_video = input("\nSave video output? (y/n): ").lower() == 'y'
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))
    
    # Crop parameters (center square)
    crop_size = min(h, w)
    center_x, center_y = w // 2, h // 2
    half_crop = crop_size // 2
    
    # For temporal smoothing
    smoothed_coords = None
    
    print("\n" + "=" * 70)
    print("Starting inference... Press 'q' to quit")
    print("=" * 70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop center region
            top = max(center_y - half_crop, 0)
            left = max(center_x - half_crop, 0)
            bottom = min(center_y + half_crop, h)
            right = min(center_x + half_crop, w)
            cropped = frame[top:bottom, left:right]
            
            # Resize for model input
            resized = cv2.resize(cropped, (128, 128))
            
            # Normalize and prepare tensor
            input_tensor = torch.from_numpy(resized).permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0).float() / 255.0
            input_tensor = input_tensor.to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)[0].cpu().numpy()
            
            # Remove low confidence
            for k in range(len(output)):
                threshold = np.max(output[k]) * 0.4
                output[k][output[k] < threshold] = 0
            
            # Extract coordinates
            scale = crop_size / output.shape[-1]
            coords = weighted_local_peak_coords(output, scale=1.0)[::-1]
            
            # Temporal smoothing
            if smoothed_coords is None:
                smoothed_coords = coords.copy()
            else:
                smoothed_coords = (
                    SMOOTHING_ALPHA * smoothed_coords +
                    (1 - SMOOTHING_ALPHA) * coords
                )
            
            # Scale to cropped frame
            x_proj = (smoothed_coords[1] * scale).astype(int)
            y_proj = (smoothed_coords[0] * scale).astype(int)
            
            # Draw skeleton
            display_frame = cropped.copy()
            draw_skeleton(
                display_frame,
                np.vstack((x_proj, y_proj)),
                (0, 255, 0)  # Green
            )
            
            # Show frame
            cv2.imshow("Hand Pose Estimation (Press 'q' to quit)", display_frame)
            
            # Save to video
            if save_video:
                # Pad to original size
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[top:bottom, left:right] = display_frame
                out.write(padded)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("Inference stopped")
        if save_video:
            print("Video saved: output.mp4")
        print("=" * 70)


if __name__ == "__main__":
    main()