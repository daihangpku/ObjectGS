import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_sam_data(file_path, index=0, prefix="sam_fb", output_dir="vis_sam"):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        print(f"Keys in file: {list(f.keys())}")
        if prefix not in f:
            # Try to guess prefix
            keys = list(f.keys())
            if len(keys) > 0:
                prefix = keys[0]
                print(f"Prefix '{prefix}' not found. Using '{keys[0]}' instead.")
            else:
                print("Error: No groups found in HDF5 file.")
                return
        
        group = f[prefix]
        
        # Check if index exists
        if str(index) not in group["pixel_level_keys"]:
             print(f"Error: Index {index} not found in {prefix}/pixel_level_keys.")
             return

        pixel_level_keys = group[f"pixel_level_keys/{index}"][...]
        scale_3d = group[f"scale_3d/{index}"][...]
        # group_cdf = group[f"group_cdf/{index}"][...]

    print(f"Loaded data for index {index}")
    print(f"Pixel level keys shape: {pixel_level_keys.shape}") # (H, W, K)
    print(f"Scale 3D shape: {scale_3d.shape}") # (Num_masks, 1)

    H, W, K = pixel_level_keys.shape
    num_total_masks = scale_3d.shape[0]
    
    # Generate random colors for each mask ID
    # Add 1 to handle potential -1 indexing if we didn't mask it out, but better to mask out.
    colors = np.random.rand(num_total_masks, 3)
    
    # Visualize each layer of masks
    for k in range(K):
        mask_indices = pixel_level_keys[:, :, k]
        
        # Create image
        vis_image = np.zeros((H, W, 3))
        
        valid_mask = mask_indices != -1
        
        # Assign colors
        # mask_indices[valid_mask] gives indices into 'colors'
        if np.any(valid_mask):
            vis_image[valid_mask] = colors[mask_indices[valid_mask]]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_image)
        plt.title(f"SAM Data - Index {index} - Layer {k} (Smallest to Largest)")
        plt.axis('off')
        
        output_path = os.path.join(output_dir, f"sam_vis_idx{index}_layer{k}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved layer {k} visualization to {output_path}")

    # Also visualize scale if possible?
    # Map scale to image
    # For layer 0
    mask_indices_0 = pixel_level_keys[:, :, 0]
    scale_image = np.zeros((H, W))
    valid_mask = mask_indices_0 != -1
    
    if np.any(valid_mask):
        # scale_3d is (N, 1)
        scales = scale_3d.flatten()
        scale_image[valid_mask] = scales[mask_indices_0[valid_mask]]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(scale_image, cmap='viridis')
    plt.colorbar(label='3D Scale')
    plt.title(f"SAM Data - Index {index} - Layer 0 Scale")
    plt.axis('off')
    output_path = os.path.join(output_dir, f"sam_vis_idx{index}_layer0_scale.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved scale visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAM Data from HDF5 file")
    parser.add_argument("file_path", type=str, help="Path to sam_data.hdf5")
    parser.add_argument("--index", type=int, default=0, help="Image index to visualize")
    parser.add_argument("--prefix", type=str, default="sam_fb", help="Prefix in hdf5 file (e.g. sam_fb)")
    parser.add_argument("--output_dir", type=str, default="vis_sam", help="Output directory for images")
    
    args = parser.parse_args()
    
    visualize_sam_data(args.file_path, args.index, args.prefix, args.output_dir)
