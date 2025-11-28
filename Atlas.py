import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
import gc # Import garbage collector module
import argparse

def generate_3d_atlas_projection(volume_input_dir, mask_input_dir, atlas_output_dir, num_samples):
    # Create output directory if it doesn't exist
    os.makedirs(atlas_output_dir, exist_ok=True)

    print(f"Volume Input directory: {volume_input_dir}")
    print(f"Mask Input directory: {mask_input_dir}")
    print(f"Atlas Output directory: {atlas_output_dir}")

    volume_nii_files = [f for f in os.listdir(volume_input_dir) if f.endswith('.nii')]

    if not volume_nii_files:
        print(f"No .nii files found in {volume_input_dir}")
        return

    eligible_files = []
    for filename in volume_nii_files:
        volume_file_path = os.path.join(volume_input_dir, filename)
        mask_file_path = os.path.join(mask_input_dir, filename)

        if not os.path.exists(mask_file_path):
            print(f"Skipping {filename}: Corresponding mask file not found at {mask_file_path}")
            continue
        try:
            volume_header = nib.load(volume_file_path)
            mask_header = nib.load(mask_file_path)
            if any(dim > 900 for dim in volume_header.shape) or any(dim > 900 for dim in mask_header.shape):
                print(f"Skipping {filename}: One or more dimensions > 900")
            else:
                eligible_files.append(filename)
            del volume_header, mask_header
            gc.collect()
        except Exception as e:
            print(f"Error reading header for {filename}: {e}")

    if not eligible_files:
        print("No eligible files found after filtering.")
        return

    selected_files = random.sample(eligible_files, min(len(eligible_files), num_samples))
    print(f"Processing {len(selected_files)} samples out of {len(eligible_files)} eligible files.")

    for filename in selected_files:
        volume_file_path = os.path.join(volume_input_dir, filename)
        mask_file_path = os.path.join(mask_input_dir, filename)

        try:
            # Process Volume
            img_vol = nib.load(volume_file_path)
            data_vol = img_vol.get_fdata()
            if data_vol.ndim == 4: data_vol = data_vol[:, :, :, 0]
            if data_vol.ndim != 3:
                print(f"Skipping {filename}: Volume not a 3D image.")
                continue
            
            data_vol = np.nan_to_num(data_vol, nan=-1024.0)
            bone_min_hu = 300
            bone_max_hu = 1000
            data_vol_windowed = np.clip((data_vol - bone_min_hu) / (bone_max_hu - bone_min_hu + 1e-8), 0, 1)

            # Process Mask
            img_mask = nib.load(mask_file_path)
            data_mask = img_mask.get_fdata()
            if data_mask.ndim == 4: data_mask = data_mask[:, :, :, 0]
            if data_mask.ndim != 3:
                print(f"Skipping {filename}: Mask not a 3D image.")
                continue

            # --- 3D Atlas Projection Logic ---
            # 1. Get indices of the max intensity values from the volume along the projection axis.
            mip_indices = np.argmax(data_vol_windowed, axis=1)
            
            # 2. Create the volume MIP using these max values.
            mip_vol = np.max(data_vol_windowed, axis=1)

            # 3. Create the mask MIP by sampling the 3D mask at the exact locations of the max intensity.
            d0, d2 = np.ogrid[:mip_indices.shape[0], :mip_indices.shape[1]]
            mip_mask = data_mask[d0, mip_indices, d2]
            
            # --- Coloring and Saving Logic (same as before) ---
            atlas_img = plt.cm.gray(mip_vol)
            unique_labels = np.unique(mip_mask)
            if len(unique_labels) > 1:
                num_labels = len(unique_labels)
                colors = plt.cm.get_cmap('tab10', num_labels)
                for i, label in enumerate(unique_labels):
                    if label == 0: continue
                    base_color_rgb = np.array(colors(i)[:3])
                    label_mask = (mip_mask == label)
                    intensity = mip_vol[label_mask]
                    shaded_color_rgb = base_color_rgb * intensity[:, np.newaxis]
                    atlas_img[label_mask, :3] = shaded_color_rgb

            output_filename_atlas = os.path.join(atlas_output_dir, f"atlas_3d_projection_{os.path.splitext(filename)[0]}.png")
            plt.imsave(output_filename_atlas, atlas_img)
            print(f"Saved 3D-projected atlas for {filename} to {output_filename_atlas}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            if 'data_vol' in locals(): del data_vol
            if 'img_vol' in locals(): del img_vol
            if 'data_mask' in locals(): del data_mask
            if 'img_mask' in locals(): del img_mask
            gc.collect()

    print(f"3D-projected atlas images generated for {len(selected_files)} files and saved to {atlas_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate 2D atlas images by projecting from a 3D atlas.")
    parser.add_argument('--input_dir', type=str, required=True, help="Base input directory with 'volumes' and 'segmentations'.")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for atlas images.")
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples to process.")
    args = parser.parse_args()

    volume_input_dir = os.path.join(args.input_dir, "volumes")
    mask_input_dir = os.path.join(args.input_dir, "segmentations")
    atlas_output_dir = os.path.join(args.output_dir, "atlas_images")

    generate_3d_atlas_projection(volume_input_dir, mask_input_dir, atlas_output_dir, args.num_samples)

if __name__ == "__main__":
    main()
