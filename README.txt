# Hand Pose Estimation with PyTorch

A deep learning model for 21-keypoint hand pose estimation using U-Net architecture and Adaptive Winged Loss, achieving **95.3% PCK@0.2 accuracy** on the FreiHAND dataset.

## Performance

- **PCK@0.2 Accuracy**: 95.3% on test set (2,560 samples)
- **Mean Pixel Error**: 3.4 pixels
- **Per-Keypoint Accuracy**: 88-99% across all 21 hand joints
- **Train/Test Gap**: 1.3% (excellent generalization)

## Architecture

- **Model**: U-Net with encoder-decoder structure
- **Loss Function**: Adaptive Winged Loss (optimized for keypoint localization)
- **Input**: RGB images (128x128)
- **Output**: 21 heatmaps(128x128) (one per keypoint)

## Dataset

Trained on [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/projects/freihand/):
- Training: 30,000 samples
- Testing: 2,560 samples (held-out)
- 21 keypoints per hand (wrist, thumb, fingers)

## Installation
```bash
# Clone repository
git clone https://github.com/DanielTheProgrammerGuy/HandPoseEstimation_PyTorch.git
cd HandPoseEstimation_PyTorch

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```python
# Download FreiHAND dataset from official source and title FreiHAND_pub_v2 and place in DATASETS folder

python Training.py
```

### Evaluation
```python
python Accuracy_Calculation.py
python Testing_Validation
```

### Inference
```python
import torch
from model import UNet2DHandPose

# Load model
model = UNet2DHandPose()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_image)
```

## Key Features

- **Adaptive Winged Loss**: Specialized loss function for accurate keypoint localization
- **Dual Coordinate Extraction**: Softmax and weighted local peak methods
- **GPU Acceleration**: 15x training speedup with CUDA
- **Robust Generalization**: Consistent accuracy on unseen data

## Results Breakdown

| Keypoint Type | Accuracy |
|--------------|----------|
| Wrist        | 98.0%    |
| Palm Joints  | 96-99%   |
| Fingertips   | 88-92%   |

Fingertips are the most challenging due to their small size and high variability.

## Technical Details

- **Framework**: PyTorch
- **Training**: NVIDIA RTX GPU with CUDA
- **Heatmap Resolution**: 128x128
- **Coordinate Extraction**: Softmax-based weighted average
- **Data Augmentation**: Rotation, scaling, brightness adjustment

## Files

- `train.py` - Training script
- `test.py` - Evaluation script
- `model.py` - U-Net architecture definition
- `Data_Extraction.py` - Data preprocessing utilities
- `Training.py` - Training loop and loss functions

## Citation

If you use this work, please cite the FreiHAND dataset:
```
@inproceedings{zimmermann2019freihand,
  title={FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

## License

MIT License

## Contact

Daniel Afonin - [afonind@mcmaster.ca](mailto:afonind@mcmaster.ca)

LinkedIn: [linkedin.com/in/daniel-afonin](https://www.linkedin.com/in/daniel-afonin-8030a038a/)