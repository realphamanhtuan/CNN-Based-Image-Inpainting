import os

from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from lib.worker import VanisherWorker
from gatedconvworker.model.gatedconvnetwork import *
from gatedconvworker.model.loss import *

class GatedConvWorker(VanisherWorker):
    def __init__(self, checkpoint_path, learning_rate):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up network
        self.generator = Generator(5).to(self.device)
        self.discriminator = Discriminator(4, 64).to(self.device)

        # optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        self.checkpoint_path = checkpoint_path
        if os.path.exists(self.checkpoint_path):
            state_dicts = torch.load(checkpoint_path)
            if 'G' in state_dicts.keys():
                self.generator.load_state_dict(state_dicts['G'])
            if 'D' in state_dicts.keys():
                self.discriminator.load_state_dict(state_dicts['D'])

            if 'G_optim' in state_dicts.keys():
                self.g_optimizer.load_state_dict(state_dicts['G_optim'])
            if 'D_optim' in state_dicts.keys():
                self.d_optimizer.load_state_dict(state_dicts['D_optim'])
            print(f"Loaded models from: {checkpoint_path}!")
        
        self.toTensorTransform = transforms.ToTensor()

    def Compute(self, gt_path, mask_path, out_path):
        return True

    def Train(self, epochCount, train_batch_size, test_batch_size, data_train, data_test, log_interval):
        train_size = len(data_train)
        test_size = len(data_test)
        print("Loaded training dataset with {} train samples, {} test samples, and {} masks".format(train_size, test_size, data_train.maskCount))

        train_iters = train_size // train_batch_size
        test_iters = test_size // test_batch_size

        for epoch in range(0, epochCount):
            print ("Training epoch", epoch, train_iters)
            self.generator.train()
            self.discriminator.train()

            # training loop
            iterator_train = iter(torch.utils.data.DataLoader(data_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True))
            for i in tqdm(range(0, train_iters)):
                torch.cuda.empty_cache()
                losses = {}
                mask, gt = next(iterator_train)

                gt = gt.to(self.device)
                #print(mask.shape)
                mask = (mask > 0.5).to(dtype=torch.float32, device=self.device)

                # 0 -> undamaged & 1 -> damaged -> (1 - mask) * gt = damaged image
                image = gt * (1.0 - mask)
                ones = torch.ones(image.shape[0], 1, image.shape[2], image.shape[3]).to(self.device)

                # generate inpainted images
                x1, x2 = self.generator(torch.cat([image, ones, mask], axis=1), mask)

                # fill the ground truth into missing holes
                final_output = x2 * mask + image * (1.0 - mask)

                pos = torch.cat((gt, mask), dim=1) #gan with mask
                neg = torch.cat((final_output.detach(), mask), dim=1) #gan with mask

                discriminator_output = self.discriminator(torch.cat((pos, neg))) #discriminator will be trained to tell fake ones from real ones and vice versa
                d_real, d_fake = torch.split(discriminator_output, train_batch_size)
                losses['d_loss'] = loss_l1(d_real, d_fake)

                self.d_optimizer.zero_grad()
                losses['d_loss'].backward()
                self.d_optimizer.step()

                x2_neg = torch.cat((x2, mask), dim=1)
                discriminator_output = self.discriminator(x2_neg)

                losses['g_loss'] = loss_sngan(discriminator_output)
                losses['g_loss'] += torch.mean((torch.abs(gt - x1))) + torch.mean((torch.abs(gt - x2)))

                self.g_optimizer.zero_grad()
                losses['g_loss'].backward()
                self.g_optimizer.step()

                del losses
                del gt, mask, image
                del ones
                del x1, x2
                del final_output
                del pos, neg
                del discriminator_output
                del d_real, d_fake
                del x2_neg

            del iterator_train

            # save state dict at the end of epoch
            state_dicts = {'G': self.generator.state_dict(), 'D': self.discriminator.state_dict(), 'G_optim': self.g_optimizer.state_dict(), 'D_optim': self.d_optimizer.state_dict(), 'n_iter': 0}
            torch.save(state_dicts, self.checkpoint_path)
            print("Saved to check point")
            del state_dicts

            print("Testing after epoch", epoch, test_iters)
            self.generator.eval()
            self.discriminator.eval()
            iterator_test = iter(torch.utils.data.DataLoader(data_test, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False))
            test_losses = {"d": 0, "g": 0}
            for i in tqdm(range(0, test_iters)):
                torch.cuda.empty_cache()
                losses = {}
                mask, gt = next(iterator_test)

                gt = gt.to(self.device)
                #print(mask.shape)
                mask = (mask > 0.5).to(dtype=torch.float32, device=self.device)

                # 0 -> undamaged & 1 -> damaged -> (1 - mask) * gt = damaged image
                image = gt * (1.0 - mask)
                ones = torch.ones(image.shape[0], 1, image.shape[2], image.shape[3]).to(self.device)

                # generate inpainted images
                x1, x2 = self.generator(torch.cat([image, ones, mask], axis=1), mask)

                # fill the ground truth into missing holes
                final_output = x2 * mask + image * (1.0 - mask)

                pos = torch.cat((gt, mask), dim=1)
                neg = torch.cat((final_output.detach(), mask), dim=1)

                discriminator_output = self.discriminator(torch.cat((pos, neg))) #discriminator will be trained to tell fake ones from real ones and vice versa
                d_real, d_fake = torch.split(discriminator_output, train_batch_size)
                losses['d_loss'] = loss_l1(d_real, d_fake)

                x2_neg = torch.cat((x2, mask), dim=1)
                discriminator_output = self.discriminator(x2_neg)

                losses['g_loss'] = loss_sngan(discriminator_output)
                losses['g_loss'] += torch.mean((torch.abs(gt - x1))) + torch.mean((torch.abs(gt - x2)))

                test_losses["d"] += losses["d_loss"].item()
                test_losses["g"] += losses["g_loss"].item()

                del losses
                del gt, mask, image
                del ones
                del x1, x2
                del final_output
                del pos, neg
                del discriminator_output
                del d_real, d_fake
                del x2_neg

            sum = 0
            for key, value in test_losses.items():
                print(key, value / test_iters)
                sum += value / test_iters
            print ("total loss:", sum)
            del iterator_test
            del test_losses

if __name__ == "__main__":
    worker = GatedConvWorker()
    worker.Loop()
    
