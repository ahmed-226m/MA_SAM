properties = {
    'nerve':{

        'weight_path': r'',

        'img_path': r'',
        'mask_path': r'',
        'atlas_path': r'',

        'data_file_path': r'',
    },
    'spine_2d': {
        'weight_path': r'd:/GP/main/phase_2/MA_SAM/utiles/ma-sam/weights',
        'img_path': r'd:/GP/main/phase_2/MA_SAM/utiles/2d_data/mip_2d_images_volumes',
        'mask_path': r'd:/GP/main/phase_2/MA_SAM/utiles/2d_data/mip_2d_images_masks',
        'atlas_path': r'd:/GP/main/phase_2/MA_SAM/utiles/2d_data/atlas_images',
        'data_file_path': r'd:/GP/main/phase_2/MA_SAM/utiles/ma-sam/data_lists', 
    },
    'SAM_weight_path': r'd:/GP/main/phase_2/MA_SAM/utiles/ma-sam/weights',
    'model_size': {
            'small': {
                'encoder_embed_dim': 768,
                'encoder_depth': 12,
                'encoder_num_heads': 12,
                'encoder_global_attn_indexes': [2, 5, 8, 11],
                'name': 'sam_vit_b_01ec64.pth'
            },
            'medium': {
                'encoder_embed_dim': 1024,
                'encoder_depth': 24,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [5, 11, 17, 23],
                'name': 'sam_vit_l_0b3195.pth'
            },
            'large': {
                'encoder_embed_dim': 1280,
                'encoder_depth': 32,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [7, 15, 23, 31],
                'name': 'sam_vit_h_4b8939.pth'
            }
        }

}
