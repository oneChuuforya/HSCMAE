import copy

import torch
import warnings

from numpy import arange

from model import encoder
from model.decoder import LightDecoder
from model import build_sparse_encoder
warnings.filterwarnings('ignore')
from args import args#, Test_data, Train_data_all, Train_data, Target_data
from dataset import Dataset
from model.HSCMAE import HSCMAE
from process import Trainer
import torch.utils.data as Data
import torch.nn as nn
import os
import json
from datautils import load_UCR, load_HAR

def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    #args:
    if args.dataset == 'ucr':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_UCR(path, folder=args.UCR_folder)
        args.num_class = len(set(Train_data[1]))
    elif args.dataset == 'har':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_HAR(path)
        args.num_class = len(set(Train_data[1]))

    if args.adjust:
        path = args.data_path
        _, Target_data, Test_data = load_UCR(path, folder=args.targetdata)
        args.num_class = len(set(Target_data[1]))
    else:
        Target_data = Train_data
    args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
    args.lr_decay_steps = args.eval_per_steps
    args.save_path = 'expnew/' + args.dataset + '/' + args.UCR_folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    config_file = open(args.save_path + '/args.json', 'w')
    tmp = args.__dict__
    json.dump(tmp, config_file, indent=1)
    print(args)
    config_file.close()

    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    target_dataset = Dataset(device=args.device, mode='supervise_train', data=Target_data, wave_len=args.wave_length)
    target_loader = Data.DataLoader(target_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    args.target_shape = target_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)


    args.sbn = True
    args.dp = 0.1

    print(args.data_shape)
    print('dataset initial ends')
    unet_layers = args.unet_layers
    depths = args.unet_depths
    dims = copy.deepcopy(args.unet_dims)
    enc: encoder.SparseEncoder = build_sparse_encoder('convnext_tiny',unet_layers=unet_layers, depths=depths, dims=dims,
                                                      bigkernel = args.bigker,input_size=args.data_shape[0], sbn=args.sbn,
                                                      drop_path_rate=args.dp, verbose=False)
    dec = LightDecoder(enc.downsample_ratio, sbn=args.sbn)
    global_dec = LightDecoder(enc.downsample_ratio, sbn=args.sbn)
    model = HSCMAE(args, unet_layers, depths, dims,
                   sparse_encoder=enc, dense_decoder=dec, global_decoder = global_dec).to(args.device)

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, train_linear_loader, test_loader,target_loader, verbose=True)

    trainer.pretrain()
    trainer.finetune()


if __name__ == '__main__':
    if args.dataset == 'har':
        args.UCR_folder = 'HAR'
        args.unet_layers = 4
        args.unet_depths = [3, 3, 6, 3]
        args.unet_dims = [96, 192, 384, 768]
    else:
        if args.UCR_folder == 'EthanolConcentration':
            args.train_batch_size = 16
            args.test_batch_size = 16
        args.dataset = 'ucr'
        args.unet_layers = 3
        args.unet_depths = [3, 6, 3]
        args.unet_dims = [96, 192, 384]
    main()