import torch
from torch.utils.data import DataLoader

from MAE_pre_train.modeling_pretrain import PretrainVisionTransformerEncoder
from train_3DRecNet.config import input_args
from train_3DRecNet.train_rec import train
from util.read_dataset import ShapeNet_voxel
from train_3DRecNet.model.Decoder_3D import _G
from train_3DRecNet.model.Discriminator import _D

args = input_args()
print('args', args)

dataset_vox = ShapeNet_voxel(args.vox_dir)

dataloader_vox = DataLoader(dataset_vox, batch_size=args.batch_size, shuffle=False)

encoder = PretrainVisionTransformerEncoder().cuda()

decoder = _G().cuda()

discriminator = _D().cuda()

loss = torch.nn.BCELoss().cuda()

optim_G = torch.optim.Adam(decoder.parameters(), lr=args.lr)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

train(args, dataloader_vox, encoder, decoder, discriminator, optim_G, optim_D, loss)
