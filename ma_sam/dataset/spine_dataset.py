import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SpineDataset(Dataset):
    def __init__(self, img_dir, mask_dir, atlas_dir, image_size=512, num_atlases=1):
        """
        Args:
            img_dir (str): Path to the directory containing target CT images (2D).
            mask_dir (str): Path to the directory containing ground truth masks.
            atlas_dir (str): Path to the directory containing atlas masks.
            image_size (int): Target size for resizing (default 512).
            num_atlases (int): Number of atlases to use per image.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.atlas_dir = atlas_dir
        self.image_size = image_size
        self.num_atlases = num_atlases

        # Scan for image files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.startswith('mip_sagittal_volume_case_') and f.endswith('.png')])
        
        # Verify matching files exist
        self.data_pairs = []
        for img_file in self.img_files:
            # Extract case ID: mip_sagittal_volume_case_0000.png -> 0000
            case_id = img_file.split('_')[-1].split('.')[0]
            
            mask_file = f'mip_sagittal_mask_case_{case_id}.png'
            atlas_file = f'atlas_3d_projection_case_{case_id}.png'
            
            if os.path.exists(os.path.join(mask_dir, mask_file)):
                # For this dataset, it seems we might use the same atlas for all, or specific ones.
                # Based on user request, we have an atlas directory. 
                # If specific atlas exists for the case, use it. 
                # If not, we might need a strategy. For now, let's assume 1-to-1 matching if exists,
                # or we pick random/all available if that's the logic.
                # Looking at file list: atlas_3d_projection_case_0005.png, etc.
                # It seems not every case has an atlas? Or maybe we use available atlases as prompts for others?
                # Standard multi-atlas usually uses *other* cases as atlases.
                # Let's collect ALL available atlases first.
                pass
            
            self.data_pairs.append({
                'img': img_file,
                'mask': mask_file,
                'case_id': case_id
            })

        # Collect all available atlases
        self.all_atlases = sorted([f for f in os.listdir(atlas_dir) if f.startswith('atlas_3d_projection_case_') and f.endswith('.png')])
        if len(self.all_atlases) == 0:
            print(f"Warning: No atlas files found in {atlas_dir}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        data_item = self.data_pairs[index]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, data_item['img'])
        img = Image.open(img_path).convert('L')
        
        # 2. Load Mask (Ground Truth)
        mask_path = os.path.join(self.mask_dir, data_item['mask'])
        if os.path.exists(mask_path):
             # Ensure ground-truth masks are single-channel
             mask = Image.open(mask_path).convert('L')
        else:
            # Fallback if mask missing? Or error?
            # For now create empty mask
            mask = Image.new('L', img.size, 0)

        # 3. Load Atlases (Prompts)
        # Strategy: Randomly select N atlases from the available pool, excluding the current case if possible (leave-one-out)
        # Or if the user meant "atlas_images" contains the specific atlas for that case?
        # Given the names "atlas_3d_projection_case_XXXX", it looks like specific projections.
        # But usually in MA-SAM, you use *other* labeled images as atlases.
        # Let's implement: Pick N random atlases from the pool.
        
        current_case_atlas_name = f"atlas_3d_projection_case_{data_item['case_id']}.png"
        possible_atlases = [a for a in self.all_atlases if a != current_case_atlas_name]
        
        if len(possible_atlases) < self.num_atlases:
            # If not enough, sample with replacement or take all
            selected_atlases = np.random.choice(self.all_atlases, self.num_atlases, replace=True)
        else:
            selected_atlases = np.random.choice(possible_atlases, self.num_atlases, replace=False)
            
        atlas_masks = []
        for atlas_name in selected_atlases:
            ap = os.path.join(self.atlas_dir, atlas_name)
            am = Image.open(ap).convert('L')
            am = am.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            am_np = np.array(am)
            atlas_masks.append(torch.from_numpy(am_np).float())
        
        if len(atlas_masks) > 0:
            atlas_tensor = torch.stack(atlas_masks, dim=0)
        else:
            # Handle case with no atlases - maybe return zeros
            atlas_tensor = torch.zeros((self.num_atlases, self.image_size, self.image_size))

        # 4. Preprocessing & Resizing
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        # 5. Convert to Tensor
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) 
        img_tensor = img_tensor.repeat(3, 1, 1) 
        img_tensor = img_tensor / 255.0 

        mask_np = np.array(mask)
        # Add a channel dimension so masks have shape [1, H, W] as expected by the loss
        mask_tensor = torch.from_numpy(mask_np).long().unsqueeze(0)

        return img_tensor, mask_tensor, atlas_tensor