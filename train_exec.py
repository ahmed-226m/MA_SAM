import argparse
import os
import sys
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_sam.model.segmentation_sub_network.MASAM import MASAM, load_from, froze
from ma_sam.model.utils import set_seed
from ma_sam.monitor.Monitor import Monitor
from ma_sam.loss_fun.CEDCLoss import CEDCLoss
from ma_sam.dataset.spine_dataset import SpineDataset
from ma_sam.trainer.Trainer import val_with_atlas_DDP_2D, train_with_atlas_DDP_2D
from ma_sam.utils.properties import properties

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables for DDP (even if using single GPU, DDP is often expected by the code structure)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9999'

def process(rank, args, train_dataset, val_dataset, weight_base_path, sam_weight_path, config):
    print(f"Process started with rank {rank}")
    set_seed(221)
    
    # Initialize Process Group
    # Select backend depending on whether CUDA/GPU is available
    # - use 'nccl' when CUDA is available (fast GPU backend)
    # - fall back to 'gloo' when no CUDA is available (CPU-friendly)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, init_method='env://', rank=rank, world_size=args.world_size)
    
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, num_workers=args.train_workers,
                                  batch_size=args.train_batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=args.val_workers,
                                batch_size=args.val_batch_size)

    # Model
    net = MASAM(args.num_classes, args.num_atlases,
                prompt_embed_dim=config.prompt_embed_dim, image_size=config.image_size,
                vit_patch_size=config.vit_patch_size, encoder_embed_dim=config.encoder_embed_dim,
                encoder_depth=config.encoder_depth, encoder_num_heads=config.encoder_num_heads,
                encoder_global_attn_indexes=config.encoder_global_attn_indexes, adapterTrain=args.adapter_train)

    epoch_start, best_dice, best_epoch, model_dic, opt_dic = 0, 0, 0, None, None
    
    # Load Weights
    if args.resume_train:
        pth_name = 'latest.pth'
        weight_path = weight_base_path + os.sep + pth_name
        if os.path.exists(weight_path):
            with open(weight_path, "rb") as f:
                dic = torch.load(f, map_location=device)
                epoch_start, best_dice, best_epoch, model_dic, opt_dic = dic['epoch'] + 1, dic['best_dice'], \
                    dic['best_epoch'], dic['model'], dic['optimizer']
                net.load_state_dict(model_dic)
            if rank==0:
                print(f'load {pth_name}, exiting best dice: {best_dice}, achieved by epoch: {best_epoch}')
        else:
            if rank==0:
                print(f'err! weight not exit! use path {weight_path}')
            # exit() # Don't exit, just start from scratch if not found
    else:
        if os.path.exists(sam_weight_path):
            with open(sam_weight_path, "rb") as f:
                state_dict = torch.load(f, map_location='cpu')

            if rank==0: print('interpolate sam-pretrained weight')
            net_new_dict = load_from(net.state_dict(), state_dict, config.image_size, config.vit_patch_size)
            net.load_state_dict(net_new_dict)
            if rank==0: print(f"load sam-pretrained weight: {sam_weight_path.split('/')[-1]}")
        else:
            if rank==0: print(f"Warning: SAM weights not found at {sam_weight_path}. Initializing randomly.")

    # DDP Wrap
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    net = DDP(net, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Monitor
    if rank == 0:
        monitor = Monitor(weight_base_path=weight_base_path, resume_train=args.resume_train, class_num=args.num_classes,
                        best_dice=best_dice, best_epoch=best_epoch, train_length=len(train_dataloader),
                        val_length=len(val_dataloader), save_every=args.save_every)
    else:
        monitor=None

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    if opt_dic is not None:
        optimizer.load_state_dict(opt_dic)

    # Loss
    cedcloss = CEDCLoss(args.num_classes).to(device)

    # Training Loop
    early_stop_flag = torch.tensor(0, device=device)
    for epoch in range(epoch_start, args.epochs):
        froze(net)
        train_with_atlas_DDP_2D(rank, net, monitor, train_sampler, train_dataloader, cedcloss, optimizer, epoch, device, args)

        val_with_atlas_DDP_2D(rank, net, monitor, val_dataloader, cedcloss, optimizer, epoch, device, args, config.image_size)

        if rank == 0:
            monitor.info_to_file(epoch=epoch, args=args.__dict__)
            if monitor.early_stop_step():
                early_stop_flag = torch.tensor(1, device=device)

        dist.barrier()
        dist.all_reduce(early_stop_flag, op=dist.ReduceOp.MAX)
        if early_stop_flag.item() > 0: break

    if rank == 0:
        monitor.end()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--num_classes', type=int, default=4, help='class num')
    parser.add_argument('--num_atlases', type=int, default=1, help='atlas num') # Changed default to 1
    parser.add_argument('--size', type=str, default='small', help='size of sam image encoder')
    parser.add_argument('--train_batch_size', type=int, default=2, help='train batch size for each gpu')
    parser.add_argument('--val_batch_size', type=int, default=2, help='val batch size for each gpu')
    parser.add_argument('--train_workers', type=int, default=0, help='train worker num') # 0 for Windows safety
    parser.add_argument('--val_workers', type=int, default=0, help='val worker num')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save_every', type=int, default=5, help='how many epochs to save model')
    parser.add_argument('--early_stop', action='store_true', help='early stop')
    parser.add_argument('--resume_train', action='store_true', help='resume train')
    parser.add_argument('--adapter_train', action='store_true', help='use adapter')
    args = parser.parse_args()
    
    print("Arguments:", args.__dict__)

    data_key = 'spine_2d'
    image_size = 512
    
    # Force single GPU/CPU for simplicity on Windows unless configured otherwise
    args.world_size = 1 
    print(f'World size: {args.world_size}')

    weight_base_path = properties[data_key]['weight_path']
    os.makedirs(weight_base_path, exist_ok=True)

    img_path = properties[data_key]['img_path']
    mask_path = properties[data_key]['mask_path']
    atlas_path = properties[data_key]['atlas_path']

    sam_weight_base_path = properties['SAM_weight_path']
    # Handle case where model_size might not be fully configured or we want defaults
    if 'model_size' in properties and args.size in properties['model_size']:
        sam_weight_name = properties['model_size'][args.size]['name']
        sam_weight_path = f'{sam_weight_base_path}/{sam_weight_name}'
        encoder_embed_dim = properties['model_size'][args.size]['encoder_embed_dim']
        encoder_depth = properties['model_size'][args.size]['encoder_depth']
        encoder_num_heads = properties['model_size'][args.size]['encoder_num_heads']
        encoder_global_attn_indexes = properties['model_size'][args.size]['encoder_global_attn_indexes']
    else:
        # Fallback defaults for 'small' (ViT-B)
        sam_weight_path = 'sam_vit_b_01ec64.pth'
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]

    # Initialize Datasets
    # We use the same directory for train and val for now, as splitting logic wasn't specified.
    # Ideally, we should split the file list.
    full_dataset = SpineDataset(img_dir=img_path, mask_dir=mask_path, atlas_dir=atlas_path, image_size=image_size, num_atlases=args.num_atlases)
    
    # Simple split: 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    config = SimpleNamespace(
        prompt_embed_dim = 256,
        vit_patch_size = 16,
        image_size = image_size,
        encoder_embed_dim = encoder_embed_dim,
        encoder_depth = encoder_depth,
        encoder_num_heads = encoder_num_heads,
        encoder_global_attn_indexes = encoder_global_attn_indexes)

    mp.spawn(process, args=(args, train_dataset, val_dataset, weight_base_path, sam_weight_path, config, ),
             nprocs=args.world_size, join=True)
