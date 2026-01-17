"""
Training script for Hand Pose Estimation U-Net model
Uses Adaptive Winged Loss and mixed precision training
"""
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as Fv2

import numpy as np

import cv2
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False)

import matplotlib.pyplot as plt

from Config import config

# Enable cuDNN benchmarking and TF32 for faster training
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2DHandPose(nn.Module):
    """
    U-Net architecture for hand pose estimation
    
    Args:
        init_channels: Number of channels in first conv block
    """
    
    def __init__(self, init_channels=None):
        super().__init__()
        if init_channels is None:
            init_channels = config["unet_channels"]
        
        N = init_channels
        
        # Encoder
        self.conv1 = ConvBlock(3, N)
        self.conv2 = ConvBlock(N, 2 * N)
        self.conv3 = ConvBlock(2 * N, 4 * N)
        self.conv4 = ConvBlock(4 * N, 8 * N)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv5 = ConvBlock(12 * N, 4 * N)  # 8N + 4N
        self.conv6 = ConvBlock(6 * N, 2 * N)   # 4N + 2N
        self.conv7 = ConvBlock(3 * N, N)       # 2N + N

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(N, 21, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Heatmaps [B, 21, H, W]
        """
        # Encoder path
        x1 = self.conv1(x)              # N x 128 x 128
        x2 = self.conv2(self.pool(x1))  # 2N x 64 x 64
        x3 = self.conv3(self.pool(x2))  # 4N x 32 x 32
        x4 = self.conv4(self.pool(x3))  # 8N x 16 x 16

        # Decoder path with skip connections
        u1 = self.up1(x4)                    # 8N x 32 x 32
        u1 = torch.cat([u1, x3], dim=1)      # 12N x 32 x 32
        u1 = self.conv5(u1)                  # 4N x 32 x 32
        
        u2 = self.up2(u1)                    # 4N x 64 x 64
        u2 = torch.cat([u2, x2], dim=1)      # 6N x 64 x 64
        u2 = self.conv6(u2)                  # 2N x 64 x 64
        
        u3 = self.up3(u2)                    # 2N x 128 x 128
        u3 = torch.cat([u3, x1], dim=1)      # 3N x 128 x 128
        u3 = self.conv7(u3)                  # N x 128 x 128

        # Output heatmaps
        out = self.final(u3)                 # 21 x 128 x 128
        out = out.clamp(min=0)
        
        # Resize if heatmap size differs from image size
        if config["heatmap_size"] != config["image_size"]:
            scale = config["heatmap_size"] / config["image_size"]
            out = F.interpolate(
                out, 
                scale_factor=scale, 
                mode="bilinear", 
                align_corners=False
            )
        
        return out


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    
    More robust than MSE for keypoint localization
    Reference: Wang et al. "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    
    Args:
        omega: Overall loss scale
        theta: Threshold between log and linear regions
        epsilon: Smoothing parameter
        alpha: Adaptivity parameter
    """
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Calculate Adaptive Wing Loss
        
        Args:
            pred: Predicted heatmaps
            target: Ground truth heatmaps
            
        Returns:
            Loss value
        """
        delta = torch.abs(pred - target)
        
        # Adaptive parameters based on ground truth
        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (
            torch.pow(self.theta / self.epsilon, self.alpha - target - 1)
        ) * (1 / self.epsilon)
        
        C = (
            self.theta * A 
            - self.omega * torch.log(
                1 + torch.pow(self.theta / self.epsilon, self.alpha - target)
            )
        )
        
        # Piecewise loss: logarithmic for small errors, linear for large
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(
                1 + torch.pow(delta / self.epsilon, self.alpha - target)
            ),
            A * delta - C
        )
        
        return loss.mean()
    

def gpu_augment(images, heatmaps):
    """
    images:   [B, 3, H, W] CUDA
    heatmaps: [B, 21, H, W] CUDA
    """
    B, _, H, W = images.shape
    device = images.device

    # ---- Random affine ----
    angles = (torch.rand(B, device=device) * 2 - 1) * 25 * torch.pi / 180
    scales = 1 + (torch.rand(B, device=device) * 2 - 1) * 0.2
    tx = (torch.rand(B, device=device) * 2 - 1) * 0.15
    ty = (torch.rand(B, device=device) * 2 - 1) * 0.15

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = scales * torch.cos(angles)
    theta[:, 0, 1] = -scales * torch.sin(angles)
    theta[:, 1, 0] = scales * torch.sin(angles)
    theta[:, 1, 1] = scales * torch.cos(angles)
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, images.size(), align_corners=False)

    images = F.grid_sample(
        images, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )

    heatmaps = F.grid_sample(
        heatmaps, grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False
    )

    # ---- Horizontal flip ----
    flip = torch.rand(B, device=device) < 0.5
    images[flip] = torch.flip(images[flip], dims=[3])
    heatmaps[flip] = torch.flip(heatmaps[flip], dims=[3])

    # ---- GPU Gaussian Blur ----
    if torch.rand(1) < 0.5:
        # Kernel size 5x5, sigma between 0.1 and 2.0
        images = Fv2.gaussian_blur(images, kernel_size=[5, 5], sigma=[0.1, 2.0])

    # ---- Color jitter (brightness only, cheap & effective) ----
    brightness = 1.0 + 0.4 * (torch.rand(B, 1, 1, 1, device=device) - 0.5)
    images = torch.clamp(images * brightness, 0, 1)

    return images, heatmaps


class AugmentedDataset(Dataset):
    """
    Dataset with on-the-fly data augmentation
    
    Args:
        images: Image tensor [N, 3, H, W]
        heatmaps: Heatmap tensor [N, 21, H, W]
        transform: Optional transform function
    """
    
    def __init__(self, images, heatmaps, transform=None):
        self.images = images
        self.heatmaps = heatmaps
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        heatmap = self.heatmaps[idx]

        if self.transform:
            img, heatmap = self.transform(img, heatmap)

        return img, heatmap


def apply_geometric_transforms(img, heatmap):
    """
    Apply random geometric transformations to both image and heatmap
    
    Args:
        img: Image tensor [3, H, W]
        heatmap: Heatmap tensor [21, H, W]
        
    Returns:
        Transformed (image, heatmap)
    """
    import random
    
    # Random affine parameters
    angle, translations, scale, shear = v2.RandomAffine.get_params(
        degrees=(-25, 25),
        translate=(0.15, 0.15),
        scale_ranges=(0.8, 1.2),
        shears=None,
        img_size=img.shape[1:]
    )

    # Apply same transform to both
    img = Fv2.affine(
        img, angle, translations, scale, shear,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0
    )
    heatmap = Fv2.affine(
        heatmap, angle, translations, scale, shear,
        interpolation=v2.InterpolationMode.NEAREST,
        fill=0
    )
    
    # Random horizontal flip
    if random.random() < 0.5:
        img = Fv2.hflip(img)
        heatmap = Fv2.hflip(heatmap)
    
    # Random Gaussian blur (image only)
    if random.random() < 0.5:
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        img = torch.from_numpy(img_np).permute(2, 0, 1)

    return img, heatmap


def apply_color_jitter(img, heatmap):
    """
    Apply color jittering to image only
    
    Args:
        img: Image tensor
        heatmap: Heatmap tensor (unchanged)
        
    Returns:
        (jittered_img, heatmap)
    """
    color_jitter = v2.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    )
    return color_jitter(img), heatmap


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """
    Train for one epoch
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, heatmaps) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        images, heatmaps = gpu_augment(images, heatmaps)


        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, heatmaps)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        print(f"Batch {batch_idx + 1}/{len(loader)}: Loss = {total_loss/(batch_idx+1):.4f}",end="\r")
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Validate model
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, heatmaps in loader:
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def save_training_curve(train_losses, val_losses, save_path):
    """Save training/validation loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Progress", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    """Main training loop"""
    print("=" * 70)
    print(" " * 20 + "Hand Pose Estimation Training")
    print("=" * 70)
    
    # Load or resume model
    load_checkpoint = input("\nLoad existing model? (y/n): ").lower() == 'y'
    checkpoint_path = None
    
    if load_checkpoint:
        models_dir = config["model_save_dir"]
        models = os.listdir(models_dir)
        
        if len(models) == 0:
            print("No models found. Training from scratch.")
            load_checkpoint = False
        else:
            print("\nAvailable models:")
            for i, model_name in enumerate(models):
                print(f"  {i}. {model_name}")
            
            idx = int(input("Enter model index: "))
            checkpoint_path = os.path.join(models_dir, models[idx])
    
    # Load data
    print("\nLoading dataset...")
    from Data_Extraction import data
    
    heatmaps = torch.from_numpy(data["heatmaps"])
    images = torch.from_numpy(data["data"]).permute(0, 3, 1, 2)
    
    # Train/validation split
    train_images = images[:30000]
    train_heatmaps = heatmaps[:30000]
    val_images = images[30000:]
    val_heatmaps = heatmaps[30000:]
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"\nDevice: {device}")
    print(f"Mixed Precision: {use_amp}")
    
    # Data augmentation
    train_transforms = v2.Compose([
        apply_geometric_transforms,
        apply_color_jitter
    ])
    
    # Create datasets
    train_dataset = AugmentedDataset(train_images, train_heatmaps, transform=None)
    val_dataset = AugmentedDataset(val_images, val_heatmaps, transform=None)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = UNet2DHandPose().to(device)
    model = torch.compile(model, backend="cudagraphs")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Load checkpoint if specified
    if load_checkpoint and checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    # Setup training
    criterion = AdaptiveWingLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=4,
    )
    scaler = GradScaler()
    
    # Training parameters
    num_epochs = config["epochs"]
    early_stop_patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_path = os.path.join(config["model_save_dir"], f"model-{timestamp}.pth")
    curve_path = f"training_curves/curve-{timestamp}.png"
    
    os.makedirs("training_curves", exist_ok=True)
    os.makedirs(config["model_save_dir"], exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, use_amp
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Time: {epoch_time:6.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save curve every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_training_curve(train_losses, val_losses, curve_path)
    
    # Final save
    save_training_curve(train_losses, val_losses, curve_path)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved: {model_path}")
    print(f"Training curve: {curve_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()