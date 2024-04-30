import copy
import torch.nn as nn

from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.embeddings import PositionalEncoding, Embeddings
from model.attention import MultiHeadedAttention
from common import Generator

class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(self.w_1(x).relu()))

class EncoderDecoder(nn.Module):
	"""
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_transformer(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
	  Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
	  Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
	  nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
	  nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
	  Generator(d_model, tgt_vocab),
	)

	# This was important from their code.
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model
