import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.augmentations import PromptUNetAugmenter

def generate_dummy_data(h=128, w=128):
    """
    Generate dummy data to test interpolation and values.
    x (image): continuous values [0, 1]
    y (mask): categorical values [0, 1]
    p (prompt): prompt image and prompt mask concatenated
    """
    # Image: Gradient to test continuous interpolation + Some grid lines
    xv, yv = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    x = (xv + yv) / 2.0
    x[::16, :] = 1.0
    x[:, ::16] = 1.0
    x = np.expand_dims(x, axis=-1).astype(np.float32)
    
    # Target Mask: A few blocks with class 1 
    y = np.zeros((h, w, 1), dtype=np.float32)
    y[30:60, 30:60, 0] = 1.0
    y[80:110, 80:110, 0] = 1.0
    
    # Prompt Image: Another gradient
    p_x = (xv * yv)
    p_x = np.expand_dims(p_x, axis=-1).astype(np.float32)
    
    # Prompt Mask: Same as y but slightly modified
    p_y = np.copy(y)
    p_y[40:50, 40:50, 0] = 0.0 # Forget some part
    
    # Combine prompt
    p = np.concatenate([p_x, p_y], axis=-1)
    
    return tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(p)

def print_statistics(name, tensor):
    arr = tensor.numpy()
    unique_vals = np.unique(arr)
    
    print(f"--- {name} ---")
    print(f"Shape: {arr.shape}")
    print(f"Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
    if len(unique_vals) <= 15:
        print(f"Unique values: {unique_vals}")
    else:
        print(f"Unique values (count): {len(unique_vals)} (Continuous data)")
    print()

def main():
    print("Generating Dummy Data...")
    x, y, p = generate_dummy_data()
    
    print("Original Data Statistics:")
    print_statistics("Original x", x)
    print_statistics("Original y", y)
    print_statistics("Original p (Image + Mask)", p)
    
    # Force 100% probabilities to make sure everything triggers
    # We want to see how values change and if interpolations introduce bad values in mask.
    augmenter_all = PromptUNetAugmenter(
        prob_photo=1.0,
        prob_gamma=1.0,
        prob_noise=1.0,
        prob_independent_noise=1.0,
        prob_morph=1.0,
        prob_dropout=1.0,
        prob_false_pos=1.0
    )
    
    print("\nApplying Augmentations (All Probs = 1.0)...")
    x_aug, y_aug, p_aug = augmenter_all(x, y, p)
    
    print("Augmented Data Statistics:")
    print_statistics("Augmented x", x_aug)
    print_statistics("Augmented y", y_aug)
    print_statistics("Augmented p", p_aug)
    
    p_x_aug, p_y_aug = tf.split(p_aug, num_or_size_splits=2, axis=-1)
    print_statistics("Augmented p_y (Prompt Mask Only)", p_y_aug)
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Augmentation Tests (Top: Original, Bottom: Augmented)", fontsize=16)
    
    p_x_orig, p_y_orig = tf.split(p, 2, axis=-1)
    
    # Row 0: Original
    axes[0, 0].imshow(np.squeeze(x), cmap='gray')
    axes[0, 0].set_title("Original x")
    axes[0, 1].imshow(np.squeeze(y), cmap='gray')
    axes[0, 1].set_title("Original y")
    axes[0, 2].imshow(np.squeeze(p_x_orig), cmap='gray')
    axes[0, 2].set_title("Original p_x")
    axes[0, 3].imshow(np.squeeze(p_y_orig), cmap='gray')
    axes[0, 3].set_title("Original p_y")
    
    # Row 1: Augmented
    axes[1, 0].imshow(np.squeeze(x_aug), cmap='gray')
    axes[1, 0].set_title("Augmented x")
    axes[1, 1].imshow(np.squeeze(y_aug), cmap='gray')
    axes[1, 1].set_title("Augmented y")
    axes[1, 2].imshow(np.squeeze(p_x_aug), cmap='gray')
    axes[1, 2].set_title("Augmented p_x")
    axes[1, 3].imshow(np.squeeze(p_y_aug), cmap='gray')
    axes[1, 3].set_title("Augmented p_y")
    
    for ax in axes.flatten():
        ax.axis('off')
        
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/augmentation_test.png")
    print("\nSaved visualization to results/augmentation_test.png")
    plt.show()

if __name__ == "__main__":
    main()
