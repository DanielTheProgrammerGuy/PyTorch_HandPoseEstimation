"""
Calculate PCK accuracy and mean error for trained model
"""
import os
import numpy as np
import torch
from Training import UNet2DHandPose
from Data_Extraction import data
from utils import softmax_coords, weighted_local_peak_coords


def calculate_pck(predicted, ground_truth, alpha=0.2):
    """
    Calculate Percentage of Correct Keypoints (PCK)
    
    Args:
        predicted: Predicted coordinates [N, 21, 2]
        ground_truth: Ground truth coordinates [N, 21, 2]
        alpha: Threshold multiplier (default 0.2)
        
    Returns:
        PCK percentage
    """
    # Calculate bounding box diagonal for normalization
    mins = ground_truth.min(axis=1)  # [N, 2]
    maxs = ground_truth.max(axis=1)  # [N, 2]
    diagonals = np.sqrt(((maxs - mins) ** 2).sum(axis=1))  # [N]
    
    # Calculate distances
    distances = np.sqrt(((predicted - ground_truth) ** 2).sum(axis=2))  # [N, 21]
    
    # Check if within threshold
    thresholds = alpha * diagonals[:, np.newaxis]  # [N, 1]
    correct = distances < thresholds
    
    return correct.mean() * 100


def calculate_mean_error(predicted, ground_truth):
    """Calculate mean pixel error across all keypoints"""
    distances = np.sqrt(((predicted - ground_truth) ** 2).sum(axis=2))
    return distances.mean()


def per_keypoint_pck(predicted, ground_truth, alpha=0.2):
    """Calculate PCK for each keypoint individually"""
    mins = ground_truth.min(axis=1)
    maxs = ground_truth.max(axis=1)
    diagonals = np.sqrt(((maxs - mins) ** 2).sum(axis=1))
    
    distances = np.sqrt(((predicted - ground_truth) ** 2).sum(axis=2))
    thresholds = alpha * diagonals[:, np.newaxis]
    correct = distances < thresholds
    
    # Average across samples for each keypoint
    return correct.mean(axis=0) * 100


def evaluate_model(model, images, heatmaps, device):
    """
    Run inference and extract coordinates
    
    Returns:
        tuple: (softmax_coords, weighted_peak_coords, ground_truth_coords)
    """
    model.eval()
    
    softmax_predictions = []
    weighted_predictions = []
    ground_truths = []
    
    print("Running inference...")
    with torch.no_grad():
        for i in range(len(images)):
            # Prepare input
            img_tensor = torch.from_numpy(
                images[np.newaxis, i, :]
            ).permute(0, 3, 1, 2)
            img_tensor = img_tensor.to(device)
            
            # Forward pass
            output = model(img_tensor)
            output_np = output.squeeze().cpu().numpy()
            
            # Remove low-confidence values
            for j in range(len(output_np)):
                threshold = np.max(output_np[j]) * 0.4
                output_np[j][output_np[j] < threshold] = 0
            
            # Scale factor from heatmap to image
            scale = img_tensor.shape[-1] / output_np.shape[-1]
            
            # Extract coordinates
            gt_coords = softmax_coords(heatmaps[i], scale).T
            pred_coords_softmax = softmax_coords(output_np, scale).T
            pred_coords_weighted = weighted_local_peak_coords(output_np, scale).T
            
            ground_truths.append(gt_coords)
            softmax_predictions.append(pred_coords_softmax)
            weighted_predictions.append(pred_coords_weighted)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(images)} samples")
    
    return (
        np.array(softmax_predictions),
        np.array(weighted_predictions),
        np.array(ground_truths)
    )


def main():
    """Main evaluation script"""
    print("=" * 70)
    print(" " * 20 + "Model Evaluation")
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
    print(f"\nLoaded model: {models[model_idx]}")
    print(f"Device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    heatmap_data = data["heatmaps"][30000:30000+2560]
    image_data = data["data"][30000:30000+2560]
    print(f"Test samples: {len(image_data)}")
    
    # Evaluate
    print("\n" + "-" * 70)
    softmax_pred, weighted_pred, ground_truth = evaluate_model(
        model, image_data, heatmap_data, device
    )

    print(softmax_pred.shape, weighted_pred.shape, ground_truth.shape)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("RESULTS - Method 1: Softmax Coordinates")
    print("=" * 70)

    per_kp_pck_softmax = per_keypoint_pck(softmax_pred.copy(), ground_truth)
    error_softmax = calculate_mean_error(softmax_pred.copy(), ground_truth)
    pck_softmax = calculate_pck(softmax_pred.copy(), ground_truth)

    print(f"PCK@0.2: {pck_softmax:.2f}%")
    print(f"Mean Error: {error_softmax:.2f} pixels")

    print("\nPer-Keypoint PCK@0.2:")
    keypoint_names = [
        'Wrist', 'Thumb1', 'Thumb2', 'Thumb3', 'Thumb4',
        'Index1', 'Index2', 'Index3', 'Index4',
        'Middle1', 'Middle2', 'Middle3', 'Middle4',
        'Ring1', 'Ring2', 'Ring3', 'Ring4',
        'Pinky1', 'Pinky2', 'Pinky3', 'Pinky4'
    ]
    
    for name, acc in zip(keypoint_names, per_kp_pck_softmax):
        print(f"  {name:12s}: {acc:5.1f}%")
    
    print(f"\nBest keypoint:  {per_kp_pck_softmax.max():.1f}%")
    print(f"Worst keypoint: {per_kp_pck_softmax.min():.1f}%")
    
    # Method 2
    print("\n" + "=" * 70)
    print("RESULTS - Method 2: Weighted Local Peak")
    print("=" * 70)

    pck_weighted = calculate_pck(weighted_pred.copy(), ground_truth)
    error_weighted = calculate_mean_error(weighted_pred.copy(), ground_truth)
    per_kp_pck_weighted = per_keypoint_pck(weighted_pred.copy(), ground_truth)


    print(f"PCK@0.2: {pck_weighted:.2f}%")
    print(f"Mean Error: {error_weighted:.2f} pixels")
    
    print("\nPer-Keypoint PCK@0.2:")
    for name, acc in zip(keypoint_names, per_kp_pck_weighted):
        print(f"  {name:12s}: {acc:5.1f}%")
    
    print(f"\nBest keypoint:  {per_kp_pck_weighted.max():.1f}%")
    print(f"Worst keypoint: {per_kp_pck_weighted.min():.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best method: {'Softmax' if pck_softmax > pck_weighted else 'Weighted Peak'}")
    print(f"Best PCK@0.2: {max(pck_softmax, pck_weighted):.2f}%")
    print(f"Best Mean Error: {min(error_softmax, error_weighted):.2f} pixels")
    print("=" * 70)

if __name__ == "__main__":
    main()