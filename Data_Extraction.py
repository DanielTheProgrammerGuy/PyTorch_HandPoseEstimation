import os
import numpy as np
import cv2
import json
from scipy.ndimage import gaussian_filter
from pathlib import Path
import time
import numpy as np
from Config import config

path = "DATASETS/FreiHAND_pub_v2"

SIZE = config["image size"]
heatmap_scale = config["heatmap size"]/config["image size"]

RADIUS = 6
SIGMA = 2

def precompute_gaussian():
    size = RADIUS*2 + 1
    coords = np.arange(size) - RADIUS
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    kernel = np.exp(-(xx**2 + yy**2) / (2 * SIGMA**2))
    return kernel

def draw_gaussian(heatmap, x, y, kernel):
    H, W = heatmap.shape
    radius = kernel.shape[0] // 2

    # Clip bounding box
    x0 = max(0, x - radius)
    x1 = min(W, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(H, y + radius + 1)

    # Kernel crop to match edges
    kx0 = radius - (x - x0)
    kx1 = kx0 + (x1 - x0)
    ky0 = radius - (y - y0)
    ky1 = ky0 + (y1 - y0)

    heatmap[y0:y1, x0:x1] = kernel[ky0:ky1, kx0:kx1]
    return heatmap


def load_dataset(subfolder):
    print(f"Loading dataset for subfolder: {subfolder}")
    data = []
    folder_path = f"{path}/{subfolder}"
    max_image = 32560
    for i in range(max_image):
        file = f"{i:08d}.jpg"
        file_path = f"{folder_path}/{file}"
        img = cv2.imread(file_path)
        if img is None:
            continue
        default_image_size = img.shape[0]
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype(np.float32)/255
        data.append(img)
        if i % 1000 == 0:
            print(f"{i}/{max_image}")
    print(f"LOADED {max_image} IMAGES AND LABELS")
    return np.array(data, dtype=np.float32), default_image_size


def project_3d_to_2d(filenamelabels, filenameK, default_image_size):
    file_path = os.path.join(path, filenamelabels)
    with open(file_path) as f:
        coord_labels = np.array(json.load(f))
    filepath = os.path.join(path, filenameK)
    with open(filepath) as f:
        K_matrices = np.array(json.load(f))
    K = K_matrices[:, None, :, :].repeat(coord_labels.shape[1], axis=1)

    X, Y, Z = coord_labels[:,:,0], coord_labels[:,:,1], coord_labels[:,:,2]
    x = (K[:,:,0,0] * (X / Z) + K[:,:,0,2])
    y = (K[:,:,1,1] * (Y / Z) + K[:,:,1,2])

    # Scale to match resized image
    x = x / default_image_size * SIZE * heatmap_scale
    y = y / default_image_size * SIZE * heatmap_scale

    return x, y


def heatmaps_from_labels(x_labels, y_labels):
    heatmap_size = int(SIZE * heatmap_scale)
    heatmaps = np.zeros((x_labels.shape[0], x_labels.shape[1], heatmap_size, heatmap_size), dtype=np.float32)
    x_labels = np.clip(x_labels, 0, heatmap_size - 1).astype(np.int32)
    y_labels = np.clip(y_labels, 0, heatmap_size - 1).astype(np.int32)
    KERNEL = (precompute_gaussian() * 1).astype(np.float32)
    for i in range(x_labels.shape[0]):
        for j in range(y_labels.shape[1]):
            heatmaps[i, j] = draw_gaussian(heatmaps[i,j],x_labels[i,j],y_labels[i,j],KERNEL)
            #heatmaps[i, j] = gaussian_filter(heatmaps[i, j], sigma=sigma, radius=radius)
            #m = np.max(heatmaps[i, j])
            #if m > 0:
            #    heatmaps[i, j] /= m
        if i % 1000 == 0:
            print(f"{i}/{x_labels.shape[0]}")
    print(f"LOADED {x_labels.shape[0]} X {x_labels.shape[1]} HEATMAPS")
    return heatmaps


data, default_image_size = load_dataset("training/rgb")
#masks, default_mask_size = load_dataset("training/mask")
#data[masks<125] = 0
x, y = project_3d_to_2d("training_xyz.json", "training_K.json", default_image_size)
heatmaps = heatmaps_from_labels(x, y)
print("SAVING TO FILE...")
data = {"heatmaps": heatmaps, "data": data}
#np.savez("Training_Data.npz", heatmaps=heatmaps, data=data, labels=(x,y))
print("DATA SUCCESFULLY EXTRACTED")