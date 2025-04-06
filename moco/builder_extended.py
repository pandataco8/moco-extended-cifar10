import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

try:
   from .metaformer_blocks import TokenMixingBlock
except ImportError:
   TokenMixingBlock = None

class MoCo(nn.Module):

   def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, token_mixer=False):

       super(MoCo, self).__init__()

       self.K = K
       self.m = m
       self.T = T

       self.token_mixer = token_mixer

       self.encoder_q = base_encoder(num_classes=dim)
       self.encoder_k = base_encoder(num_classes=dim)


       if mlp:
           dim_mlp = self.encoder_q.fc.weight.shape[1]
           self.encoder_q.fc = nn.Sequential(
               nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
           self.encoder_k.fc = nn.Sequential(
               nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

       if token_mixer:
           if TokenMixingBlock is None:
               raise RuntimeError("TokenMixingBlock not available.")
           print("=> Token mixer enabled: injecting between encoder and projection head.")
           self.token_mixer_q = TokenMixingBlock(dim)
           self.token_mixer_k = TokenMixingBlock(dim)
       else:
           self.token_mixer_q = None
           self.token_mixer_k = None

       for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
           param_k.data.copy_(param_q.data)
           param_k.requires_grad = False

       self.register_buffer("queue", torch.randn(dim, K))
       self.queue = nn.functional.normalize(self.queue, dim=0)

       self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


   @torch.no_grad()
   def _momentum_update_key_encoder(self):
       """
       Momentum update of the key encoder
       """
       for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
           param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


   @torch.no_grad()
   def _dequeue_and_enqueue(self, keys):

       keys = concat_all_gather(keys)

       batch_size = keys.shape[0]


       ptr = int(self.queue_ptr)
       assert self.K % batch_size == 0

       self.queue[:, ptr:ptr + batch_size] = keys.T
       ptr = (ptr + batch_size) % self.K


       self.queue_ptr[0] = ptr


   def forward(self, im_q, im_k):

       q = self.encoder_q(im_q)

       if self.token_mixer_q:
           q = self.token_mixer_q(q)

       q = nn.functional.normalize(q, dim=1)

       with torch.no_grad():
           self._momentum_update_key_encoder()


           k = self.encoder_k(im_k)

           if self.token_mixer_k:
               k = self.token_mixer_k(k)

           k = nn.functional.normalize(k, dim=1)

       l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
       l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

       logits = torch.cat([l_pos, l_neg], dim=1)

       logits /= self.T

       labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

       self._dequeue_and_enqueue(k)

       return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):

   if not torch.distributed.is_available():
       return tensor
   if not torch.distributed.is_initialized():
       return tensor

   tensors_gather = [torch.zeros_like(tensor)
                     for _ in range(torch.distributed.get_world_size())]
   torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

   output = torch.cat(tensors_gather, dim=0)
   return output



