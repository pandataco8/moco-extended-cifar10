import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from moco.loader_extended import get_augmented_loader
from moco.builder_extended import MoCo
from models.custom_backbones import get_custom_backbone
from moco.metaformer_blocks import TokenMixingBlock


def parse_args():
   parser = argparse.ArgumentParser(description='MoCo Extended Training')
   parser.add_argument('--arch', default='resnet18', help='model architecture')
   parser.add_argument('--workers', default=4, type=int)
   parser.add_argument('--epochs', default=200, type=int)
   parser.add_argument('--batch-size', default=256, type=int)
   parser.add_argument('--lr', default=0.03, type=float)
   parser.add_argument('--momentum', default=0.999, type=float)
   parser.add_argument('--weight-decay', default=1e-4, type=float)
   parser.add_argument('--moco-dim', default=128, type=int)
   parser.add_argument('--moco-k', default=65536, type=int)
   parser.add_argument('--moco-m', default=0.999, type=float)
   parser.add_argument('--moco-t', default=0.07, type=float)
   parser.add_argument('--mlp', action='store_true', help='use MLP head')
   parser.add_argument('--token-mixer', action='store_true', help='Enable MetaFormer-style token mixing')
   parser.add_argument('--augmentation-type', default='default', choices=['default', 'aggressive'])
   parser.add_argument('--log-file', default='/content/train_log.txt')
   parser.add_argument('--visualize-aug', action='store_true', help='Visualize a few augmentation samples and exit')
   return parser.parse_args()


def main():
   args = parse_args()

   print("=> Using architecture:", args.arch)
   print("=> Queue Size:", args.moco_k, " | Momentum:", args.moco_m, " | Batch Size:", args.batch_size)
   print("=> Token Mixer Enabled:", args.token_mixer)

   os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
   log_f = open(args.log_file, 'a')
   log_f.write(f"Starting training with arch={args.arch}, queue={args.moco_k}, momentum={args.moco_m}, bs={args.batch_size}, mixer={args.token_mixer}, aug={args.augmentation_type}\n")
   log_f.flush()

   print("=> creating MoCo model")
   if args.arch in models.__dict__:
       base_encoder = models.__dict__[args.arch]
   else:
       base_encoder = get_custom_backbone(args.arch)

   model = MoCo(
       base_encoder,
       args.moco_dim, args.moco_k, args.moco_m, args.moco_t,
       mlp=args.mlp,
       token_mixer=args.token_mixer
   )

   model = model.cuda()
   model = torch.nn.DataParallel(model)
   optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
   train_loader = get_augmented_loader(args.batch_size, args.augmentation_type, workers=args.workers)

   if args.visualize_aug:
       import matplotlib.pyplot as plt


       images, _ = next(iter(train_loader))
       fig, axs = plt.subplots(2, 5, figsize=(12, 5))
       for i in range(5):
           axs[0, i].imshow(images[0][i].permute(1, 2, 0))
           axs[0, i].set_title("View 1")
           axs[0, i].axis('off')


           axs[1, i].imshow(images[1][i].permute(1, 2, 0))
           axs[1, i].set_title("View 2")
           axs[1, i].axis('off')


       plt.suptitle(f"Augmentation Type: {args.augmentation_type}")
       plt.tight_layout()
       plt.show()
       return

   for epoch in range(args.epochs):
       model.train()

       epoch_loss = 0
       for i, (images, _) in enumerate(train_loader):
           images[0] = images[0].cuda(non_blocking=True)
           images[1] = images[1].cuda(non_blocking=True)

           output, target = model(im_q=images[0], im_k=images[1])
           loss = nn.CrossEntropyLoss()(output, target)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           epoch_loss += loss.item()

       avg_loss = epoch_loss / len(train_loader)
       log_f.write(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}\n")
       log_f.flush()
       print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}")

   log_f.close()
   torch.save(model.module.encoder_q.state_dict(), "/content/encoder_q.pth")
   print("✅ Saved trained encoder_q to /content/encoder_q.pth")


if __name__ == '__main__':
   main()



