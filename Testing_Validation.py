from Training import UNet2DHandPose
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import numpy as np
import torch
import random
import os
from Data_Extraction import data

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


# Define the joint chains (FreiHAND format)
chains = [
    [0, 1, 2, 3, 4],  # Thumb
    [0, 5, 6, 7, 8],  # Index
    [0, 9, 10, 11, 12],  # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20],  # Pinky
]


def draw_skeleton(ax, coords, color, label):
    for chain in chains:
        xs = coords[1, chain]
        ys = coords[0, chain]
        ax.plot(xs, ys, color=color, marker='o', markersize=4, linewidth=2)
    ax.set_title(label)
    ax.axis('off')


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

for i in range(random_pos, NUM_SAMPLES + random_pos):
    input_tensor = torch.from_numpy(image_data[np.newaxis, i, :]).permute(0, 3, 1, 2)

    with torch.no_grad():
        output = model(input_tensor)
    output_np = output.squeeze().numpy()

    #removes extraneous small values from heatmap
    for k in range(len(output_np)):
        output_np[k][output_np[k]<np.max(output_np[k])*0.4] = 0

    scale = input_tensor.shape[-1] / output_np.shape[-1]

    real_coordinates = softmax(heatmap_data[i])
    estimated_coordinates_1 = softmax(output_np)
    estimated_coordinates_2 = weighted_local_peak(output_np)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axs:
        ax.imshow(image_data[i], origin='upper')
    draw_skeleton(axs[0], real_coordinates, 'red', "Real Keypoints")
    draw_skeleton(axs[1], estimated_coordinates_1, 'blue', "Estimated 1")
    draw_skeleton(axs[2], estimated_coordinates_2, 'green', "Estimated 2")
    plt.tight_layout()
    plt.savefig(f"Test_Results/Image/{i - random_pos}.png", dpi=200)
    plt.close(fig)

    for j in range(0):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(heatmap_data[i][j], origin='upper')
        axs[0].set_title("Ground Truth Heatmap")
        axs[0].plot(real_coordinates[1][j] / scale,
                    real_coordinates[0][j] / scale,
                    'ro', markersize=6)
        axs[0].plot(estimated_coordinates_1[1][j] / scale,
                    estimated_coordinates_1[0][j] / scale,
                    'bo', markersize=6)
        axs[0].plot(estimated_coordinates_2[1][j] / scale,
                    estimated_coordinates_2[0][j] / scale,
                    'go', markersize=6)
        axs[0].axis('off')
        axs[1].imshow(output_np[j], origin='upper')
        axs[1].set_title("Predicted Heatmap")
        axs[1].plot(real_coordinates[1][j] / scale,
                    real_coordinates[0][j] / scale,
                    'ro', markersize=6)
        axs[1].plot(estimated_coordinates_1[1][j] / scale,
                    estimated_coordinates_1[0][j] / scale,
                    'bo', markersize=6)
        axs[1].plot(estimated_coordinates_2[1][j] / scale,
                    estimated_coordinates_2[0][j] / scale,
                    'go', markersize=6)
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"Test_Results/Heatmap/{i - random_pos}_{j}.png", dpi=200)
        plt.close(fig)

    print(f"Processed sample {i + 1} ({i + 1 - random_pos}/{NUM_SAMPLES})")