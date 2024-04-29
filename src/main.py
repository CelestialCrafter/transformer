import torch

from model.transformer import make_transformer
from src.common import subsequent_mask

# sorry if this is rushed, i have to get this finished fast soo.. clean up comes later

def inference():
	model = make_transformer(11, 11, 2)
	model.eval()
	src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
	src_mask = torch.ones(1, 1, 10)

	memory = model.encode(src, src_mask)
	ys = torch.zeros(1, 1).type_as(src)

	for i in range(9):
		out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.data[0]
		ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)

	print("Example Untrained Model Prediction:", ys)

for _ in range(10):
	inference()
