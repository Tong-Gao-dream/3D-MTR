import torch


def train(args, dataloader_vox, unet3d, optim, loss):
    for epoch in range(args.epoch):
        for i_batch, sample in enumerate(dataloader_vox):
            sample_vox = sample.cuda()
            optim.zero_grad()
            output = unet3d(sample_vox)

            loss_ = loss(output, sample_vox)

            optim.zero_grad()
            loss_.backward()
            optim.step()
    unet3d.eval()
    torch.save(unet3d.state_dict(), './model_save/Repair_Net.pth')
