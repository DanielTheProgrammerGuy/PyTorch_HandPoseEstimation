"""
Visualize model predictions on validation samples
"""
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from Training import UNet2DHandPose
from Data_Extraction import data
from utils import softmax_coords, weighted_local_peak_coords


# Hand skeleton structure (FreiHAND format)
SKELETON_CHAINS = [
    [0, 1, 2, 3, 4],      # Thumb
    [0, 5, 6, 7, 8],      # Index
    [0, 9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20],  # Pinky
]


def draw_skeleton(ax, coords, color, label):
    """
    Draw hand skeleton on matplotlib axis
    
    Args:
        ax: Matplotlib axis
        coords: Coordinates [2, 21] (row, col)
        color: Line color
        label: Plot label
    """
    for chain in SKELETON_CHAINS:
        xs = coords[1, chain]
        ys = coords[0, chain]
        ax.plot(xs, ys, color=color, marker='o', markersize=5, linewidth=2)
    
    ax.set_title(label, fontsize=12)
    ax.axis('off')


def visualize_predictions(model, images, heatmaps, start_idx, num_samples, device):
    """
    Generate visualization of predictions vs ground truth
    
    Args:
        model: Trained model
        images: Image data
        heatmaps: Ground truth heatmaps
        start_idx: Starting index
        num_samples: Number of samples to visualize
        device: torch device
    """
    model.eval()
    
    output_dir = "Test_Results/Images"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations for {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(start_idx, start_idx + num_samples):
            # Prepare input
            img_tensor = torch.from_numpy(
                images[np.newaxis, i, :]
            ).permute(0, 3, 1, 2).to(device)
            
            # Forward pass
            output = model(img_tensor)
            output_np = output.squeeze().cpu().numpy()
            
            # Remove low confidence predictions
            for j in range(len(output_np)):
                threshold = np.max(output_np[j]) * 0.4
                output_np[j][output_np[j] < threshold] = 0
            
            # Scale factor
            scale = img_tensor.shape[-1] / output_np.shape[-1]
            
            # Extract coordinates
            gt_coords = softmax_coords(heatmaps[i], scale)
            pred_softmax = softmax_coords(output_np, scale)
            pred_weighted= weighted_local_peak_coords(output_np, scale)
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for ax in axes:
                ax.imshow(images[i], origin='upper')
            
            draw_skeleton(axes[0], gt_coords, 'red', "Ground Truth")
            draw_skeleton(axes[1], pred_softmax, 'blue', "Softmax Prediction")
            draw_skeleton(axes[2], pred_weighted, 'green', "Weighted Peak Prediction")
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"sample_{i-start_idx:04d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if (i - start_idx + 1) % 5 == 0:
                print(f"  Saved {i - start_idx + 1}/{num_samples} visualizations")

    print(f"Visualizations saved to: {output_dir}")


def main():
    """Main visualization script"""
    print("=" * 70)
    print(" " * 15 + "Validation Sample Visualization")
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
    print(f"Loaded model: {models[model_idx]}")

    # Load validation data
    print("\nLoading validation data...")
    heatmap_data = data["heatmaps"][30000:32560]
    image_data = data["data"][30000:32560]

    # Random starting point
    num_samples = 5
    max_start = len(image_data) - num_samples
    start_idx = random.randint(0, max_start)

    print(f"\nGenerating {num_samples} visualizations starting from index {start_idx}...")

    # Generate visualizations
    visualize_predictions(
        model, image_data, heatmap_data,
        775, num_samples, device
    )

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
if __name__ == "__main__":
    main()
