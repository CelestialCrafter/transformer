import torch
from common import Batch

def synthetic(V, batch_size, nbatches):
	"Generate random data for a src-tgt copy task."
	for i in range(nbatches):
		data = torch.randint(1, V, size=(batch_size, 10))
		data[:, 0] = 1
		src = data.requires_grad_(False).clone().detach()
		tgt = data.requires_grad_(False).clone().detach()
		yield Batch(src, tgt, 0)
