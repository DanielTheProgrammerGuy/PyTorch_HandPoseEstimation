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

##Detailed result Breakdown (output from Accuracy_Calculation)

======================================================================
RESULTS - Method 1: Softmax Coordinates
======================================================================
PCK@0.2: 96.74%
Mean Error: 2.91 pixels

Per-Keypoint PCK@0.2:
  Wrist       :  99.1%
  Thumb1      :  98.2%
  Thumb2      :  97.7%
  Thumb3      :  96.4%
  Thumb4      :  90.8%
  Index1      :  98.3%
  Index2      :  97.5%
  Index3      :  96.6%
  Index4      :  92.1%
  Middle1     :  99.4%
  Middle2     :  98.9%
  Middle3     :  97.7%
  Middle4     :  94.1%
  Ring1       :  98.6%
  Ring2       :  98.7%
  Ring3       :  97.7%
  Ring4       :  94.0%
  Pinky1      :  98.1%
  Pinky2      :  97.9%
  Pinky3      :  96.7%
  Pinky4      :  93.1%

Best keypoint:  99.4%
Worst keypoint: 90.8%

======================================================================
RESULTS - Method 2: Weighted Local Peak
======================================================================
PCK@0.2: 96.46%
Mean Error: 2.99 pixels

Per-Keypoint PCK@0.2:
  Wrist       :  98.7%
  Thumb1      :  97.8%
  Thumb2      :  97.5%
  Thumb3      :  96.0%
  Thumb4      :  90.6%
  Index1      :  98.6%
  Index2      :  97.2%
  Index3      :  95.9%
  Index4      :  91.8%
  Middle1     :  99.2%
  Middle2     :  98.4%
  Middle3     :  97.3%
  Middle4     :  93.4%
  Ring1       :  98.6%
  Ring2       :  98.7%
  Ring3       :  97.2%
  Ring4       :  93.4%
  Pinky1      :  98.1%
  Pinky2      :  97.7%
  Pinky3      :  96.5%
  Pinky4      :  93.1%

Best keypoint:  99.2%
Worst keypoint: 90.6%

======================================================================
SUMMARY
======================================================================
Best method: Softmax
Best PCK@0.2: 96.74%
Best Mean Error: 2.91 pixels
======================================================================

## Technical Details

- **Framework**: PyTorch
- **Training**: NVIDIA RTX GPU with CUDA
- **Heatmap Resolution**: 128x128
- **Coordinate Extraction**: Softmax-based weighted average
- **Data Augmentation**: Rotation, scaling, brightness adjustment

## Files

- `Training.py` - Training script and model definition
- `Testing_Validation.py` - Evaluation of Validation Visualisation script
- `Testing_Camera.py` - Evaluation of Camera Feed Visualisation script
- `Accuracy_Calculation.py` - Accuracy PCK calculations
- `Data_Extraction.py` - Data preprocessing utilities
- `utils.py` - Coordinates from heatmaps functions
- `Config.py` - Configurable Parameters

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