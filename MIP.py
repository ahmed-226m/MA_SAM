import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
import gc # Import garbage collector module
import argparse

def process_files(volume_input_dir, mask_input_dir, volume_output_dir, mask_output_dir, num_samples):
    # Create output directories if they don't exist
    os.makedirs(volume_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    print(f"Volume Input directory: {volume_input_dir}")
    print(f"Mask Input directory: {mask_input_dir}")
    print(f"Volume Output directory: {volume_output_dir}")
    print(f"Mask Output directory: {mask_output_dir}")

    # Get all .nii files from the volume input directory
    volume_nii_files = [f for f in os.listdir(volume_input_dir) if f.endswith('.nii')]

    if not volume_nii_files:
        print(f"No .nii files found in {volume_input_dir}")
    else:
        # Filter files based on slice count in any dimension before selecting samples
        eligible_files = []
        for filename in volume_nii_files:
            volume_file_path = os.path.join(volume_input_dir, filename)
            mask_file_path = os.path.join(mask_input_dir, filename) # Assume mask has same filename

            # Check if corresponding mask file exists
            if not os.path.exists(mask_file_path):
                print(f"Skipping {filename}: Corresponding mask file not found at {mask_file_path}")
                continue

            try:
                # Load only the header to check dimensions for both volume and mask
                volume_header = nib.load(volume_file_path)
                mask_header = nib.load(mask_file_path)

                # Check if any dimension is greater than 900 for either volume or mask
                if any(dim > 900 for dim in volume_header.shape) or any(dim > 900 for dim in mask_header.shape):
                    print(f"Skipping {filename}: One or more dimensions (Volume: {volume_header.shape}, Mask: {mask_header.shape}) > 900")
                else:
                    eligible_files.append(filename)

                # Explicitly delete header objects to free memory
                del volume_header
                del mask_header
                gc.collect()

            except Exception as e:
                print(f"Error reading header for {filename}: {e}")

        if not eligible_files:
            print("No eligible files found after filtering by slice count and mask existence.")
        else:
            # Select up to num_samples randomly from eligible files
            selected_files = random.sample(eligible_files, min(len(eligible_files), num_samples))
            print(f"Processing {len(selected_files)} samples out of {len(eligible_files)} eligible files.")

            for filename in selected_files:
                # Process Volume
                volume_file_path = os.path.join(volume_input_dir, filename)
                try:
                    img_vol = nib.load(volume_file_path)
                    data_vol = img_vol.get_fdata()

                    if data_vol.ndim == 4:
                        data_vol = data_vol[:, :, :, 0]
                    if data_vol.ndim != 3:
                        print(f"Skipping volume {filename}: Not a 3D image. Dimensions: {data_vol.ndim}")
                    else:
                        # get_fdata() already applies scaling, so we can use the data directly.
                        # The manual scaling that was here before was redundant and causing errors.

                        # --- Debugging and NaN handling ---
                        print(f"\n--- Debugging Volume: {filename} ---")
                        has_nans = np.isnan(data_vol).any()
                        # Use np.nanmin/nanmax to get min/max ignoring NaNs for a more accurate report
                        print(f"HU data before windowing - Min: {np.nanmin(data_vol):.2f}, Max: {np.nanmax(data_vol):.2f}, Contains NaN: {has_nans}")
                        if has_nans:
                            print("Replacing NaN values with a low value (-1024) to prevent errors.")
                            data_vol = np.nan_to_num(data_vol, nan=-1024.0)
                        # --- End of debugging ---

                        # Windowing for bone (e.g., 300 to 1000 HU)
                        bone_min_hu = 300
                        bone_max_hu = 1000
                        data_vol_windowed = (data_vol - bone_min_hu) / (bone_max_hu - bone_min_hu + 1e-8)
                        data_vol_windowed = np.clip(data_vol_windowed, 0, 1)
                        
                        print(f"Data after windowing (0-1 scale) - Min: {np.min(data_vol_windowed):.2f}, Max: {np.max(data_vol_windowed):.2f}")

                        mip_vol = np.max(data_vol_windowed, axis=1) # Sagittal MIP

                        print(f"MIP data before scaling to 255 - Min: {np.min(mip_vol):.2f}, Max: {np.max(mip_vol):.2f}")
                        print("-------------------------------------\n")
                        
                        mip_vol = (mip_vol * 255).astype(np.uint8)
                        output_filename_vol = os.path.join(volume_output_dir, f"mip_sagittal_volume_{os.path.splitext(filename)[0]}.png")
                        plt.imsave(output_filename_vol, mip_vol, cmap='gray')
                        print(f"Saved sagittal MIP for volume {filename} to {output_filename_vol}")

                except Exception as e:
                    print(f"Error processing volume {filename}: {e}")
                finally:
                    if 'data_vol' in locals():
                        del data_vol
                    if 'img_vol' in locals():
                        del img_vol
                    if 'data_vol_windowed' in locals():
                        del data_vol_windowed
                    gc.collect()

                # Process Mask
                mask_file_path = os.path.join(mask_input_dir, filename)
                try:
                    img_mask = nib.load(mask_file_path)
                    data_mask = img_mask.get_fdata()

                    if data_mask.ndim == 4:
                        data_mask = data_mask[:, :, :, 0]
                    if data_mask.ndim != 3:
                        print(f"Skipping mask {filename}: Not a 3D image. Dimensions: {data_mask.ndim}")
                    else:
                        # Masks are typically binary or integer labels, so direct normalization to 0-255 is usually fine for visualization
                        mip_mask = np.max(data_mask, axis=1) # Sagittal MIP
                        mip_mask = (mip_mask - mip_mask.min()) / (mip_mask.max() - mip_mask.min() + 1e-8)
                        mip_mask = (mip_mask * 255).astype(np.uint8)
                        output_filename_mask = os.path.join(mask_output_dir, f"mip_sagittal_mask_{os.path.splitext(filename)[0]}.png")
                        plt.imsave(output_filename_mask, mip_mask, cmap='gray')
                        print(f"Saved sagittal MIP for mask {filename} to {output_filename_mask}")

                except Exception as e:
                    print(f"Error processing mask {filename}: {e}")
                finally:
                    if 'data_mask' in locals():
                        del data_mask
                    if 'img_mask' in locals():
                        del img_mask
                    gc.collect() # Trigger garbage collection after processing both volume and mask

            print(f"Sagittal MIP 2D projections generated for {len(selected_files)} volumes and masks and saved to {volume_output_dir} and {mask_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate Maximum Intensity Projections (MIPs) for 3D medical images.")
    parser.add_argument('--input_dir', type=str, required=True, help="Base input directory containing 'volumes' and 'segmentations' subdirectories.")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory to save the MIP images.")
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples to process.")
    args = parser.parse_args()

    volume_input_dir = os.path.join(args.input_dir, "volumes")
    mask_input_dir = os.path.join(args.input_dir, "segmentations")

    volume_output_dir = os.path.join(args.output_dir, "mip_2d_images_volumes")
    mask_output_dir = os.path.join(args.output_dir, "mip_2d_images_masks")

    process_files(volume_input_dir, mask_input_dir, volume_output_dir, mask_output_dir, args.num_samples)

if __name__ == "__main__":
    main()

# python MIP.py --input_dir <your_base_input_directory> --output_dir <your_base_output_directory> --num_samples 20