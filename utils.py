"""
Utility functions for coordinate extraction from heatmaps
"""
import numpy as np
from skimage.feature import peak_local_max


def softmax_coords(heatmap, scale=1.0):
    """
    Extract coordinates using softmax (weighted average)
    
    Args:
        heatmap: Input heatmap [21, H, W]
        scale: Scale factor
        
    Returns:
        Coordinates [2, 21] (row, col)
    """
    row_indices = np.sum(
        heatmap * np.indices(heatmap.shape)[1], 
        axis=(1, 2)
    ) / (np.sum(heatmap, axis=(1, 2)) + 1e-8)
    
    col_indices = np.sum(
        heatmap * np.indices(heatmap.shape)[2], 
        axis=(1, 2)
    ) / (np.sum(heatmap, axis=(1, 2)) + 1e-8)
    
    coordinates = np.vstack([row_indices, col_indices]) * scale
    return coordinates


def max_peak_coords(heatmap, scale=1.0):
    """
    Extract coordinates using argmax
    
    Args:
        heatmap: [21, H, W]
        scale: Scale factor
        
    Returns:
        Coordinates [2, 21]
    """
    coordinates = []
    for j in range(heatmap.shape[0]):
        max_idx = np.argmax(heatmap[j])
        row, col = np.unravel_index(max_idx, heatmap[j].shape)
        coordinates.append([row * scale, col * scale])
    
    return np.array(coordinates).T


def weighted_local_peak_coords(heatmap, scale=1.0):
    """
    Extract coordinates using weighted local peak
    
    Args:
        heatmap: [21, H, W]
        scale: Scale factor
        
    Returns:
        Coordinates [2, 21]
    """
    softmax_pred = softmax_coords(heatmap, scale=1.0)
    coordinates = []
    
    for i in range(heatmap.shape[0]):
        local_peaks = peak_local_max(heatmap[i])
        
        if len(local_peaks) == 0:
            # Use softmax as fallback
            coord = [softmax_pred[0, i], softmax_pred[1, i]]
        else:
            # Find closest peak to softmax
            dists = np.sum(
                (local_peaks - softmax_pred[:, i]) ** 2,
                axis=-1
            )
            closest = np.argmin(dists)
            coord = local_peaks[closest]
        
        coordinates.append(coord)
    
    return np.array(coordinates).T * scale