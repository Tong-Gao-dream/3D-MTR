import torch
from util.var_or_cuda import var_or_cuda

def train(args, dataloader_vox, encoder, decoder, discriminator, optim_G, optim_D, loss):
    for epoch in range(args.epoch):
        for i_batch, sample in enumerate(dataloader_vox):
            # update discriminator

            sample_vox = var_or_cuda(sample)
            real_labels = var_or_cuda(torch.ones(args.batch_size))
            fake_labels = var_or_cuda(torch.zeros(args.batch_size))

            d_real = discriminator(X)
            d_real = d_real.squeeze()
            d_real_loss = loss(d_real, real_labels)


            encoder_output = encoder(sample_vox)
            fake = decoder(encoder_output)

            d_fake = discriminator(fake)
            d_fake = d_fake.squeeze()
            d_fake = d_fake.squeeze()
            d_fake_loss = loss(d_fake, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()
            # update generator
            encoder_output = encoder(sample_vox)
            fake = decoder(encoder_output)
            d_fake = discriminator(fake)
            d_fake = d_fake.squeeze()
            g_loss = loss(d_fake, real_labels)
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

    decoder.eval()
    torch.save(decoder.state_dict(), 'model_save/decoder.pth')
    discriminator().eval()
    torch.save(discriminator.state_dict(), 'model_save/discriminator.pth')
