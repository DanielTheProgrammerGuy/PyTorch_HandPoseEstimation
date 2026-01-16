"""
Configuration file for Hand Pose Estimation model
"""

config = {
    # Image/Heatmap dimensions
    "image_size": 128,
    "heatmap_size": 128,
    
    # Training hyperparameters
    "batch_size": 24,
    "learning_rate": 0.001,
    "epochs": 200,
    
    # Model architecture
    "unet_channels": 64,
    
    # Data paths
    "dataset_path": "DATASETS/FreiHAND_pub_v2",
    "model_save_dir": "models/",
    "results_dir": "Test_Results/",
}