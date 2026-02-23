import numpy as np

# --- SECOND Dataset Constants ---
SECOND_MEAN_A = np.array([113.40, 114.08, 116.45])
SECOND_STD_A  = np.array([48.30,  46.27,  48.14])

SECOND_MEAN_B = np.array([111.07, 114.04, 118.18])
SECOND_STD_B  = np.array([49.41,  47.01,  47.94])

# --- LandsatSCD Dataset Constants ---
LANDSAT_MEAN_A = np.array([141.53, 139.20, 137.73])
LANDSAT_STD_A  = np.array([81.99,  83.31,  83.89])

LANDSAT_MEAN_B = np.array([137.36, 136.50, 135.14])
LANDSAT_STD_B  = np.array([85.97,  86.01,  86.81])

def get_constants(dataset_name, time_step):
    d_name = dataset_name.lower()
    time = time_step.upper()
    
    if 'second' in d_name:
        if time == 'A': return SECOND_MEAN_A, SECOND_STD_A
        if time == 'B': return SECOND_MEAN_B, SECOND_STD_B
        
    elif 'landsat' in d_name:
        if time == 'A': return LANDSAT_MEAN_A, LANDSAT_STD_A
        if time == 'B': return LANDSAT_MEAN_B, LANDSAT_STD_B
    
    # Fallback to ImageNet
    print(f"Warning: Unknown dataset '{dataset_name}'. Using standard ImageNet normalization.")
    imagenet_mean = np.array([123.675, 116.28, 103.53])
    imagenet_std = np.array([58.395, 57.12, 57.375])
    return imagenet_mean, imagenet_std

def normalize_image(image, time_step, dataset_name):
    """
    Normalizes an image array using constants from the specified dataset.
    Returns float32 image.
    """
    mean, std = get_constants(dataset_name, time_step)
        
    image = image.astype(np.float32)
    normalized_image = (image - mean) / std
    
    return normalized_image

def denormalize_image(normalized_image, time_step, dataset_name):
    """
    Denormalizes an image array to get back original pixel values.
    Returns uint8 image.
    """
    mean, std = get_constants(dataset_name, time_step)
        
    image = (normalized_image * std) + mean
    return np.clip(image, 0, 255).astype(np.uint8)