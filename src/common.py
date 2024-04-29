import torch
import copy
import torch.nn as nn
from torch.nn.functional import log_softmax

class SublayerConnection(nn.Module):
	"""
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

class Generator(nn.Module):
	"Define standard linear + softmax generation step."

	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return log_softmax(self.proj(x), dim=-1)

def repeat(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
	return subsequent_mask == 0

class Batch:
	"""Object for holding a batch of data with mask during training."""

	def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if tgt is not None:
			self.tgt = tgt[:, :-1]
			self.tgt_y = tgt[:, 1:]
			self.tgt_mask = self.make_std_mask(self.tgt, pad)
			self.ntokens = (self.tgt_y != pad).data.sum()

	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
		return tgt_mask
