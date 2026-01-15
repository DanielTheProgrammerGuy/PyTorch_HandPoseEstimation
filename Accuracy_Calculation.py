from Training import UNet2DHandPose
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import numpy as np
import torch
import random
import os
from Data_Extraction import data


def calculate_pck(predicted_keypoints, ground_truth_keypoints, alpha=0.2):
    """
    Calculate PCK@alpha for hand pose estimation

    Args:
        predicted_keypoints: (N, 21, 2) - N samples, 21 keypoints, (x, y)
        ground_truth_keypoints: (N, 21, 2) - ground truth positions
        alpha: threshold multiplier (default 0.2)

    Returns:
        pck: Percentage of Correct Keypoints
    """

    # Calculate bounding box diagonal for each sample
    def get_bbox_diagonal(keypoints):
        # keypoints shape: (N, 21, 2)
        mins = keypoints.min(axis=1)  # (N, 2) - min x, min y
        maxs = keypoints.max(axis=1)  # (N, 2) - max x, max y
        bbox_size = maxs - mins  # (N, 2) - width, height
        diagonal = np.sqrt((bbox_size ** 2).sum(axis=1))  # (N,)
        return diagonal

    # Get normalization factor (bbox diagonal)
    diagonals = get_bbox_diagonal(ground_truth_keypoints)  # (N,)

    # Calculate Euclidean distance between predicted and ground truth
    distances = np.sqrt(((predicted_keypoints - ground_truth_keypoints) ** 2).sum(axis=2))  # (N, 21)

    # Threshold for each sample
    thresholds = alpha * diagonals[:, np.newaxis]  # (N, 1)

    # Check if each keypoint is within threshold
    correct = distances < thresholds  # (N, 21) boolean array

    # Calculate PCK (percentage across all keypoints and samples)
    pck = correct.mean() * 100

    return pck

def softmax(heatmap):
    coordinates = np.append(
        np.sum(heatmap * np.indices(heatmap.shape)[1], axis=(1, 2)) / np.sum(heatmap, axis=(1, 2))[np.newaxis, :],
        np.sum(heatmap * np.indices(heatmap.shape)[2], axis=(1, 2)) / np.sum(heatmap, axis=(1, 2))[np.newaxis, :]
        , axis=0) * scale
    return coordinates

def max_peak(heatmap):
    coordinates = []
    for j in range(heatmap.shape[0]):
        max_temp_flat = np.argmax(heatmap[j])
        row, col = np.unravel_index(max_temp_flat, heatmap[j].shape)
        row *= scale
        col *= scale
        coordinates.append((row, col))
    return np.array(coordinates).T

def weighted_local_peak(heatmap):
    softmax_coords = softmax(heatmap)
    coordinates = []
    for i in range(len(softmax_coords.T)):
        local_peaks = peak_local_max(heatmap[i])
        min_distance_index = np.argmin(np.sum(np.pow(local_peaks - softmax_coords.T[i],2),axis=-1))
        coordinates.append(local_peaks[min_distance_index])
    return np.array(coordinates).T

# Scale data/heatmaps to 0-1 range when loading
random_pos = random.randint(30000, 32560)
print("Loading Data...")
#data = np.load("Training_Data.npz")
print("loaded from file")
# Scale data/heatmaps to 0-1 range when loading
heatmap_data = data["heatmaps"]
image_data = data["data"]

model = UNet2DHandPose()
for i in range(len(os.listdir("models/"))):
    print(f"{i}. {os.listdir('models/')[i]}")
model_name = os.listdir('models/')[5]#[int(input("Enter model index: "))]
model.load_state_dict(torch.load(f"models/{model_name}", map_location=torch.device('cpu')))

NUM_SAMPLES = 10
predictions1 = []
predictions2 = []
ground_truths = []


for i in range(0, 2560):
    input_tensor = torch.from_numpy(image_data[np.newaxis, i, :]).permute(0, 3, 1, 2)

    with torch.no_grad():
        output = model(input_tensor)
    output_np = output.squeeze().numpy()

    #removes extraneous small values from heatmap
    for k in range(len(output_np)):
        output_np[k][output_np[k]<np.max(output_np[k])*0.4] = 0

    scale = input_tensor.shape[-1] / output_np.shape[-1]

    real_coordinates = softmax(heatmap_data[i]).T
    estimated_coordinates_1 = softmax(output_np).T
    estimated_coordinates_2 = weighted_local_peak(output_np).T
    ground_truths.append(real_coordinates)
    predictions1.append(estimated_coordinates_1)
    predictions2.append(estimated_coordinates_2)

    if i % 100 == 0:
        print(f"{i}. sample done")

predictions1 = np.array(predictions1)
predictions2 = np.array(predictions2)
ground_truths = np.array(ground_truths)


# Calculate mean error
def calculate_mean_error(predicted, ground_truth):
    distances = np.sqrt(((predicted - ground_truth) ** 2).sum(axis=2))
    return distances.mean()


mean_error1 = calculate_mean_error(predictions1, ground_truths)
mean_error2 = calculate_mean_error(predictions2, ground_truths)


# Calculate per-keypoint accuracy
def per_keypoint_pck(predicted, ground_truth, alpha=0.2):
    def get_bbox_diagonal(keypoints):
        mins = keypoints.min(axis=1)
        maxs = keypoints.max(axis=1)
        diagonal = np.sqrt(((maxs - mins) ** 2).sum(axis=1))
        return diagonal

    diagonals = get_bbox_diagonal(ground_truth)
    distances = np.sqrt(((predicted - ground_truth) ** 2).sum(axis=2))
    thresholds = alpha * diagonals[:, np.newaxis]
    correct = distances < thresholds
    per_keypoint = correct.mean(axis=0) * 100  # Average across samples, per keypoint
    return per_keypoint


per_kp_pck = per_keypoint_pck(predictions1, ground_truths)

print(f"\nOverall Metrics 1:")
print(f"PCK@0.2: {calculate_pck(predictions1, ground_truths):.2f}%")
print(f"Mean Error: {mean_error1:.2f} pixels")

print(f"\nPer-Keypoint PCK@0.2:")
keypoint_names = ['Wrist', 'Thumb1', 'Thumb2', 'Thumb3', 'Thumb4',
                  'Index1', 'Index2', 'Index3', 'Index4',
                  'Middle1', 'Middle2', 'Middle3', 'Middle4',
                  'Ring1', 'Ring2', 'Ring3', 'Ring4',
                  'Pinky1', 'Pinky2', 'Pinky3', 'Pinky4']

for i, (name, acc) in enumerate(zip(keypoint_names, per_kp_pck)):
    print(f"  {name}: {acc:.1f}%")

print(f"\nBest keypoints: {per_kp_pck.max():.1f}%")
print(f"Worst keypoints: {per_kp_pck.min():.1f}%")


per_kp_pck = per_keypoint_pck(predictions2, ground_truths)

print(f"\n\n\nOverall Metrics 2:")
print(f"PCK@0.2: {calculate_pck(predictions2, ground_truths):.2f}%")
print(f"Mean Error: {mean_error1:.2f} pixels")

print(f"\nPer-Keypoint PCK@0.2:")

for i, (name, acc) in enumerate(zip(keypoint_names, per_kp_pck)):
    print(f"  {name}: {acc:.1f}%")

print(f"\nBest keypoints: {per_kp_pck.max():.1f}%")
print(f"Worst keypoints: {per_kp_pck.min():.1f}%")