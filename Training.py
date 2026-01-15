import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
import torchvision.transforms.v2.functional as Fv2
import time
import random
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import os
from Config import config

torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    from Data_Extraction import data
    NUM_EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch size"]
    start_run = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def apply_color_jitter_to_image(img, heatmap):
    # Define the color jitter transform here
    color_jitter = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
    # Apply to image, return original heatmap
    return color_jitter(img), heatmap


def apply_transforms_to_both(img, heatmap):
    # Define affine ranges
    degrees = 25
    translate = (0.15, 0.15)
    scale_ranges = (0.8, 1.2)
    shears = None
    flip = random.random() < 0.5
    blur = random.random() < 0.5
    perspective_warp = random.random() < 0
    distortion = 0.7

    # Sample random affine parameters
    angle, translations, scale, shear = v2.RandomAffine.get_params(
        degrees=(-degrees, degrees),
        translate=translate,
        scale_ranges=scale_ranges,
        shears=shears,
        img_size=img.shape[1:]  # (H, W)
    )

    # Apply SAME affine transform
    img = Fv2.affine(
        img,
        angle=angle,
        translate=translations,
        scale=scale,
        shear=shear,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0)

    heatmap = Fv2.affine(
        heatmap,
        angle=angle,
        translate=translations,
        scale=scale,
        shear=shear,
        interpolation=v2.InterpolationMode.NEAREST,
        fill=0)
    
    
    if flip:
        img = Fv2.hflip(img)
        heatmap = Fv2.hflip(heatmap)
    if blur:
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        img = torch.from_numpy(img_np).permute(2, 0, 1)
    if perspective_warp:
        pts1 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        pts2 = pts1 + np.random.uniform(-distortion, distortion, pts1.shape).astype(np.float32)

        h, w = img.shape[1:]
        pts1 *= [w, h]
        pts2 *= [w, h]

        M = cv2.getPerspectiveTransform(pts1, pts2)

        img_np = img.permute(1, 2, 0).numpy()
        heat_np = heatmap.squeeze(0).permute(1, 2, 0).numpy()
        img_np = cv2.warpPerspective(img_np, M, (w, h))
        heat_np = cv2.warpPerspective(heat_np, M, (w, h))
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        heatmap = torch.from_numpy(heat_np).permute(2, 0, 1)

    return img, heatmap


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2DHandPose(nn.Module):
    def __init__(self, init_ch=config["Unet Channels"]):
        super().__init__()
        N = init_ch
        # Encoder
        self.conv1 = ConvBlock(3, N)
        self.conv2 = ConvBlock(N, 2 * N)
        self.conv3 = ConvBlock(2 * N, 4 * N)
        self.conv4 = ConvBlock(4 * N, 8 * N)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv5 = ConvBlock(8 * N + 4 * N, 4 * N)
        self.conv6 = ConvBlock(4 * N + 2 * N, 2 * N)
        self.conv7 = ConvBlock(2 * N + N, N)

        # Final head
        self.final = nn.Sequential(
            nn.Conv2d(N, 21, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x) #                3 x 128 x 128       ->      N x 128 x 128
        x2 = self.conv2(self.pool(x1)) #    N x 128 x 128       ->      2N x 64 x 64
        x3 = self.conv3(self.pool(x2)) #    2N x 64 x 64        ->      4N x 32 x 32
        x4 = self.conv4(self.pool(x3)) #    4N x 32 x 32        ->      8N x 16 x 16

        # Decoder
        u1 = self.up1(x4) #                  8N x 16 x 16        ->      8N x 32 x 32
        u1 = torch.cat([u1, x3], dim=1) #   8N+4N x 32 x 32     ->      12N x 32 x 32
        u1 = self.conv5(u1)#                12N x 32 x 32       ->      4N x 32 x 32
        u2 = self.up2(u1) #                  4N x 32 x 32        ->      4N x 64 x 64
        u2 = torch.cat([u2, x2], dim=1)#    4N+2N x 64 x 64     ->      6N x 64 x 64
        u2 = self.conv6(u2)#                6N x 64 x 64        ->      2N x 64 x 64
        u3 = self.up3(u2)#                   2N x 64 x 64        ->      2N x 128 x 128
        u3 = torch.cat([u3, x1], dim=1)#    2N+1N x 128 x 128   ->      3N x 128 x 128
        u3 = self.conv7(u3)#                3N x 128 x 128      ->      1N x 128 x 128

        # currently N x 128 x 128 -> 21 x 128 x 128
        out = self.final(u3)
        out = out.clamp(min=0)
        out = F.interpolate(out, scale_factor=config["heatmap size"]/config["image size"], mode="bilinear", align_corners=False)
        return out


class AugmentedTensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        self.data = data_tensor
        self.targets = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        heatmap = self.targets[index]

        if self.transform:
            img, heatmap = self.transform(img, heatmap)

        return img, heatmap

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()

        # Logarithmic part for small errors
        # (self.alpha - y) makes it adaptive to the heatmap ground truth
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y))) * (self.alpha - y) * (torch.pow(self.theta / self.epsilon, self.alpha - y - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y))
        
        losses = torch.where(
            delta_y < self.theta,
            self.omega * torch.log(1 + torch.pow(delta_y / self.epsilon, self.alpha - y)),
            A * delta_y - C
        )
        return losses.mean()

def main():
    # Ask to load last model
    load_last_model = input("Load Model? (y/n): ") == "y"
    if load_last_model:
        for i in range(len(os.listdir("models/"))):
            print(f"{i}. {os.listdir('models/')[i]}")
        model_name = os.listdir('models/')[int(input("Enter model index: "))]
    torch_heatmaps = torch.from_numpy(data["heatmaps"])
    torch_img = torch.from_numpy(data["data"]).permute(0, 3, 1, 2)
    print("Converted to Torch and Normalized")
    train_heatmaps = torch_heatmaps[:30000]
    train_data = torch_img[:30000]
    print("Train Data Extracted")
    test_heatmaps = torch_heatmaps[30000:]
    test_data = torch_img[30000:]
    print("Val Data Extracted")

    print(
        f"\rData Loaded  Heatmaps:[{data['heatmaps'].nbytes / 1000000000:.3f}GB] Images:[{data['data'].nbytes / 1000000000:.3f}GB]")

    '''
    i = 0
    while True:
        show_pair("title",test_data[i],test_heatmaps[i])
        i += 1
        '''

    # Setup device and AMP scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()
    print(f"Using the device: {device}")

    train_transforms = v2.Compose([
        apply_transforms_to_both,
        apply_color_jitter_to_image
    ])

    val_transforms = None  # Already preprocessed to 0-1 range

    train_dataset = AugmentedTensorDataset(train_data, train_heatmaps, transform=train_transforms)
    val_dataset = AugmentedTensorDataset(test_data, test_heatmaps, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=2,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet2DHandPose().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Params:", total_params)
    if load_last_model:
        model.load_state_dict(torch.load(f"models/{model_name}"))
    criterion = AdaptiveWingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning rate"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
    early_stop_epochs = 10
    epochs_since_improvement = 0
    best_val_loss = None

    '''
    ###
    ### TESTING
    ###
    start = time.time()
    for i, (img, heat) in enumerate(train_loader):
        if i == 100: break
    print("100 batches time:", time.time() - start)

    dummy_img = torch.randn(32, 3, 64, 64).cuda()
    dummy_heat = torch.randn(32, 21, 64, 64).cuda()

    start = time.time()
    for _ in range(200):
        out = model(dummy_img)
    torch.cuda.synchronize()
    print("GPU only time:", time.time() - start)

    ###
    ###
    ###
    '''

    train_losses = []
    test_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: ", end="")
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            bar_length = 50
            print(
                f"\rEpoch {(epoch + 1):>3}/{NUM_EPOCHS} | Epoch Runtime: {time.time() - start_time:>7.3f}s | Loss = {running_loss / (batch_idx + 1):.8f} | Progress[\033[34m{int(batch_idx / (len(train_loader) - 1) * bar_length) * '█'}\033[90m{(bar_length - int(batch_idx / (len(train_loader) - 1) * bar_length)) * '█'}\033[0m]{batch_idx + 1}/{len(train_loader)}",
                end="")

        epoch_train_loss = running_loss / (batch_idx + 1)
        print(f"\rEpoch {(epoch + 1):>3}/{NUM_EPOCHS} | Epoch Runtime: {time.time() - start_time:>7.3f}s | Loss = {epoch_train_loss:.8f}",end="")
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f" | Val_Loss = {avg_val_loss:.8f}{' '*bar_length}")
        scheduler.step(avg_val_loss)
        test_losses.append(avg_val_loss)

        if best_val_loss == None or avg_val_loss < best_val_loss:
            epochs_since_improvement = 0
            torch.save(model.state_dict(), f"models/model-{start_run}.pth")
            best_val_loss = avg_val_loss
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement > early_stop_epochs or epoch == NUM_EPOCHS - 1:
                print("Stopped imporiving, ending early!")
                torch.save(model.state_dict(), f"models/model-{start_run}.pth")
                plt.figure(figsize=(8, 5))
                plt.plot(train_losses[1:], label="Train Loss")
                plt.plot(test_losses[1:], label="Test Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Learning Curve")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"training_curves/learning_curve_{start_run}.png", dpi=200)
                break
        if (epoch+1) % 5 == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses[1:], label="Train Loss")
            plt.plot(test_losses[1:], label="Test Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Learning Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"training_curves/learning_curve_{start_run}.png", dpi=200)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()