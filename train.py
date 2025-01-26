import argparse
import torch
from logger.logger import *
from datetime import datetime
import os
import numpy as np
import random
from trainer import Trainer
from pathlib import Path
from dataset import XKDataset
from loss import CFDLoss
from model import AutoSam
from segment_anything import sam_model_registry

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=114, help="random seed")
    parser.add_argument("--save_dir", type=str, default="./save_dir", help="save dir")
    parser.add_argument("--data_dir", type=str, default="./data_dir", help="dataset dir")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=bool, default=True, help="learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_checkpoints/sam_vit_b_01ec64.pth",
                        help="sam checkpoint")
    parser.add_argument("--encoder_adapter", type=bool, default=False, help="encoder adapter")
    parser.add_argument("--class_num", type=int, default=6, help="class_num")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--run_id", type=str, default=None, help="run id")
    parser.add_argument("--save_period", type=int, default=50, help="save model period")
    parser.add_argument("--early_stop", type=int, default=400, help="early stop")
    parser.add_argument("--max_num_save", type=int, default=20, help="max num save")
    parser.add_argument("--device", type=str, default='cuda:0', help="local rank")
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    save_dir = Path(args.save_dir)
    if args.run_id is None:
        args.run_id = datetime.now().strftime(r'%m%d_%H%M')
    args.log_save_dir = save_dir / args.run_id / 'logs'
    args.model_save_dir = save_dir / args.run_id / 'models'
    os.makedirs(args.log_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    with open(args.log_save_dir / 'config.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args.__dict__.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    logger = get_logger('train', args.log_save_dir)

    device = torch.device(args.device)

    sam_model = sam_model_registry[args.model_type](args).to(device)
    model = AutoSam(args, sam_model).to(device)
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    logger.info('Model: {}'.format(model.__class__.__name__))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = CFDLoss()
    if args.lr_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)
    else:
        scheduler = None

    train_dataset = XKDataset(args.data_dir, domain='train', image_size=(args.image_size, args.image_size))
    val_dataset = XKDataset(args.data_dir, domain='val', image_size=(args.image_size, args.image_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    trainer = Trainer(args, logger, model, optimizer, criterion, device, train_loader, val_loader, scheduler)
    trainer.train()



if __name__ == '__main__':
    args = parse_args()
    main(args)