import torch
from torch.optim.lr_scheduler import LambdaLR

from data.generators import synthetic
from inference.decode import greedy_decode
from model.transformer import make_transformer
from training.loss import SimpleLossCompute
from training.regularization import LabelSmoothing
from training.loop import run_epoch
from training.optim_sched import rate, DummyOptimizer, DummyScheduler

# sorry if this is rushed, i have to get this finished fast soo.. clean up comes later

V = 11
batch_size = 256
epochs = 1

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_transformer(V, V, N=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = LambdaLR(
  optimizer=optimizer,
  lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400),
)

max_steps = -1
for epoch in range(epochs):
	model.train()
	_, _, new_max_steps = run_epoch(
	  synthetic(V, batch_size, 20),
	  model,
	  SimpleLossCompute(model.generator, criterion),
	  optimizer,
	  lr_scheduler,
		max_steps=max_steps,
		epoch=epoch,
	  mode="train",
	)

	if new_max_steps > max_steps:
		max_steps = new_max_steps

	model.eval()
	run_epoch(
	  synthetic(V, batch_size, 5),
	  model,
	  SimpleLossCompute(model.generator, criterion),
	  DummyOptimizer(),
	  DummyScheduler(),
	  mode="eval",
	)[0]

model.eval()
src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
max_len = src.shape[1]
src_mask = torch.ones(1, 1, max_len)

# @TODO get real data, tokenization
print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))
