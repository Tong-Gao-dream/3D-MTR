import torch
from torch.utils.data import DataLoader

from train_RepairNet.model.U_Net_3D import unet3d
from train_RepairNet.config import input_args
from train_RepairNet.train_repair import train
from util.read_dataset import ShapeNet_voxel

args = input_args()
print('args', args)

dataset_vox = ShapeNet_voxel(args.vox_dir)

dataloader_vox = DataLoader(dataset_vox, batch_size=args.batch_size, shuffle=False)

unet3d = unet3d().cuda()

optim = torch.optim.Adam(unet3d.parameters(), lr=args.lr)

loss = torch.nn.MSELoss().cuda()

train(args, dataloader_vox, unet3d, optim, loss)

