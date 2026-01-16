"""
Data extraction and preprocessing for FreiHAND dataset
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from Config import config

# Constants from config
SIZE = config["image_size"]
HEATMAP_SIZE = config["heatmap_size"]
DATASET_PATH = config["dataset_path"]
HEATMAP_SCALE = HEATMAP_SIZE / SIZE

# Gaussian heatmap parameters
RADIUS = 6
SIGMA = 2


def precompute_gaussian():
    """
    Precompute Gaussian kernel for heatmap generation
    
    Returns:
        np.ndarray: Gaussian kernel
    """
    size = RADIUS * 2 + 1
    coords = np.arange(size) - RADIUS
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    kernel = np.exp(-(xx**2 + yy**2) / (2 * SIGMA**2))
    return kernel


def draw_gaussian(heatmap, x, y, kernel):
    """
    Draw Gaussian blob on heatmap at specified coordinates
    
    Args:
        heatmap: Target heatmap array
        x, y: Center coordinates
        kernel: Precomputed Gaussian kernel
        
    Returns:
        Modified heatmap
    """
    H, W = heatmap.shape
    radius = kernel.shape[0] // 2

    # Clip bounding box to image boundaries
    x0 = max(0, x - radius)
    x1 = min(W, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(H, y + radius + 1)

    # Adjust kernel crop to match edges
    kx0 = radius - (x - x0)
    kx1 = kx0 + (x1 - x0)
    ky0 = radius - (y - y0)
    ky1 = ky0 + (y1 - y0)

    heatmap[y0:y1, x0:x1] = kernel[ky0:ky1, kx0:kx1]
    return heatmap


def load_images(subfolder, max_images=32560):
    """
    Load and preprocess RGB images from dataset
    
    Args:
        subfolder: Subfolder name (e.g., 'training/rgb')
        max_images: Maximum number of images to load
        
    Returns:
        tuple: (images array, original image size)
    """
    print(f"Loading images from: {subfolder}")
    images = []
    folder_path = os.path.join(DATASET_PATH, subfolder)
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    
    for i in range(max_images):
        file_path = os.path.join(folder_path, f"{i:08d}.jpg")
        
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read image {file_path}")
            continue
            
        original_size = img.shape[0]
        
        # Resize and normalize
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype(np.float32) / 255.0
        images.append(img)
        
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{max_images} images")
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images, dtype=np.float32), original_size


def project_3d_to_2d(labels_file, K_file, original_size):
    """
    Project 3D hand keypoints to 2D image coordinates
    
    Args:
        labels_file: JSON file with 3D coordinates
        K_file: JSON file with camera intrinsics
        original_size: Original image dimensions
        
    Returns:
        tuple: (x_coords, y_coords) in heatmap space
    """
    # Load 3D coordinates
    labels_path = os.path.join(DATASET_PATH, labels_file)
    with open(labels_path, 'r') as f:
        coord_3d = np.array(json.load(f))
    
    # Load camera intrinsics
    K_path = os.path.join(DATASET_PATH, K_file)
    with open(K_path, 'r') as f:
        K_matrices = np.array(json.load(f))
    
    # Repeat K for all keypoints
    K = K_matrices[:, None, :, :].repeat(coord_3d.shape[1], axis=1)
    
    # Project to 2D
    X, Y, Z = coord_3d[:, :, 0], coord_3d[:, :, 1], coord_3d[:, :, 2]
    x = K[:, :, 0, 0] * (X / Z) + K[:, :, 0, 2]
    y = K[:, :, 1, 1] * (Y / Z) + K[:, :, 1, 2]
    
    # Scale to match resized image
    x = x / original_size * HEATMAP_SIZE
    y = y / original_size * HEATMAP_SIZE
    
    return x, y


def generate_heatmaps(x_coords, y_coords):
    """
    Generate Gaussian heatmaps from 2D keypoint coordinates
    
    Args:
        x_coords: X coordinates for all samples and keypoints
        y_coords: Y coordinates for all samples and keypoints
        
    Returns:
        np.ndarray: Heatmaps array [N, 21, H, W]
    """
    n_samples = x_coords.shape[0]
    n_keypoints = x_coords.shape[1]
    
    heatmaps = np.zeros(
        (n_samples, n_keypoints, HEATMAP_SIZE, HEATMAP_SIZE),
        dtype=np.float32
    )
    
    # Clip coordinates to valid range
    x_coords = np.clip(x_coords, 0, HEATMAP_SIZE - 1).astype(np.int32)
    y_coords = np.clip(y_coords, 0, HEATMAP_SIZE - 1).astype(np.int32)
    
    kernel = precompute_gaussian().astype(np.float32)
    
    for i in range(n_samples):
        for j in range(n_keypoints):
            heatmaps[i, j] = draw_gaussian(
                heatmaps[i, j],
                x_coords[i, j],
                y_coords[i, j],
                kernel
            )
        
        if (i + 1) % 1000 == 0:
            print(f"Generated heatmaps for {i + 1}/{n_samples} samples")
    
    print(f"Successfully generated heatmaps for {n_samples} samples")
    return heatmaps


def main():
    """Main data extraction pipeline"""
    print("=" * 60)
    print("FreiHAND Dataset Extraction")
    print("=" * 60)
    
    # Load images
    images, original_size = load_images("training/rgb")
    
    # Project 3D to 2D
    print("\nProjecting 3D keypoints to 2D...")
    x, y = project_3d_to_2d(
        "training_xyz.json",
        "training_K.json",
        original_size
    )
    
    # Generate heatmaps
    print("\nGenerating Gaussian heatmaps...")
    heatmaps = generate_heatmaps(x, y)
    
    # Save data
    print("\nSaving processed data...")
    data = {
        "heatmaps": heatmaps,
        "data": images
    }
    
    print("\n" + "=" * 60)
    print("Data extraction complete!")
    print(f"Images: {images.shape}")
    print(f"Heatmaps: {heatmaps.shape}")
    print("=" * 60)
    
    return data


data = main()