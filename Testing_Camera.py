import cv2
import torch
import numpy as np
from Training import UNet2DHandPose
import os
from skimage.feature import peak_local_max


def softmax(heatmap):
    coordinates = np.append(
        np.sum(heatmap * np.indices(heatmap.shape)[1], axis=(1, 2)) / np.sum(heatmap, axis=(1, 2))[np.newaxis, :],
        np.sum(heatmap * np.indices(heatmap.shape)[2], axis=(1, 2)) / np.sum(heatmap, axis=(1, 2))[np.newaxis, :]
        , axis=0)
    return coordinates

def weighted_local_peak(heatmap):
    softmax_coords = softmax(heatmap)
    coordinates = []
    for i in range(len(softmax_coords.T)):
        local_peaks = peak_local_max(heatmap[i])
        min_distance_index = np.argmin(np.sum(np.pow(local_peaks - softmax_coords.T[i],2),axis=-1))
        coordinates.append(local_peaks[min_distance_index])
    return np.array(coordinates).T

def max_peak(heatmap):
    coordinates = []
    for j in range(heatmap.shape[0]):
        max_temp_flat = np.argmax(heatmap[j])
        row, col = np.unravel_index(max_temp_flat, heatmap[j].shape)
        coordinates.append((row, col))
    return np.array(coordinates).T

# Define skeleton drawing
chains = [
    [0, 1, 2, 3, 4],      # Thumb
    [0, 5, 6, 7, 8],      # Index
    [0, 9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20],  # Pinky
]

alpha = 0.5

def draw_skeleton(frame, coords, color, label):
    for chain in chains:
        cv2.polylines(frame, [np.array([(int(coords[1, j]), int(coords[0, j])) for j in chain])], False, color, 4)

# Load model
model = UNet2DHandPose()
for i in range(len(os.listdir("models/"))):
    print(f"{i}. {os.listdir('models/')[i]}")
model_name = os.listdir('models/')[int(input("Enter model index: "))]
model.load_state_dict(torch.load(f"models/{model_name}",map_location=torch.device('cpu')))
model.eval()

# Open webcam (0 = default)
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

# Get original frame dimensions
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")
h, w, _ = frame.shape
print(f"Camera feed size: {w}x{h}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# Define center crop box
crop_size = h
center_x, center_y = w // 2, h // 2
half = crop_size // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Crop the center 512×512 region ---
    top = max(center_y - half, 0)
    left = max(center_x - half, 0)
    bottom = min(center_y + half, h)
    right = min(center_x + half, w)
    cropped = frame[top:bottom, left:right]
    # --- Resize to 128×128 for model input ---
    resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
    resized_view = cv2.resize(cropped, (bottom-top, bottom-top), interpolation=cv2.INTER_AREA)

    frame = cropped
    left, top, right, bottom = 0, 0, 0, 0

    # --- Normalize + move channels for model ---
    input_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        output = model(input_tensor)[0].numpy()

    for k in range(len(output)):
        output[k][output[k]<np.max(output[k])*0.4] = 0

    # --- Compute estimated joint coordinates (64×64 heatmaps) ---
    indices = np.indices(output.shape)
    estimated_coordinates_1 = weighted_local_peak(output)[::-1]
    estimated_coordinates_2 = softmax(output)[::-1]
    estimated_coordinates_3 = max_peak(output)[::-1]

    # Convert from heatmap (64×64) → cropped (512×512) → full frame
    scale_factor = crop_size / output.shape[-1]  # 8.0

    if 'smoothed_coords' not in globals():
        smoothed_coords = estimated_coordinates_1.copy()

    smoothed_coords = alpha * smoothed_coords + (1 - alpha) * estimated_coordinates_1

    x_coords_1, y_coords_1 = smoothed_coords
    x_proj_1 = (x_coords_1 * scale_factor + left).astype(int)
    y_proj_1 = (y_coords_1 * scale_factor + top).astype(int)

    x_coords_2, y_coords_2 = estimated_coordinates_2
    x_proj_2 = (x_coords_2 * scale_factor + left).astype(int)
    y_proj_2 = (y_coords_2 * scale_factor + top).astype(int)

    x_coords_3, y_coords_3 = estimated_coordinates_3
    x_proj_3 = (x_coords_3 * scale_factor + left).astype(int)
    y_proj_3 = (y_coords_3 * scale_factor + top).astype(int)

    frame1 = resized_view.copy()
    frame2 = resized_view.copy()
    frame3 = resized_view.copy()

    # --- Draw points ---
    for (x, y) in zip(x_proj_1, y_proj_1):
        cv2.circle(frame1, (x, y), 10, (255, 0, 0), -1)
    for (x, y) in zip(x_proj_2, y_proj_2):
        cv2.circle(frame2, (x, y), 10, (255, 0, 0), -1)
    for (x, y) in zip(x_proj_3, y_proj_3):
        cv2.circle(frame3, (x, y), 10, (255, 0, 0), -1)

    # --- Draw lines ---
    draw_skeleton(frame1, np.vstack((y_proj_1, x_proj_1)), (100, 0, 0), "Estimated Pose")
    draw_skeleton(frame2, np.vstack((y_proj_2, x_proj_2)), (100, 0, 0), "Smoothed Pose")
    draw_skeleton(frame3, np.vstack((y_proj_3, x_proj_3)), (100, 0, 0), "Max Peak Pose")

    combined_frame = np.append(frame1, frame2, axis=0)
    combined_frame = np.append(combined_frame, frame3, axis=0)

    cv2.imshow("Hand Pose Estimation", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(combined_frame)

out.release()
cap.release()
cv2.destroyAllWindows()