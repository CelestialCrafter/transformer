import time
import os

class TrainState:
	"""Track number of steps, examples, and tokens processed"""

	step: int = 0  # Steps in the current epoch
	accum_step: int = 0  # Number of gradient accumulation steps
	samples: int = 0  # total # of examples used
	tokens: int = 0  # total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
		epoch=0,
		max_steps=-1,
    accum_iter=1,
    train_state=TrainState(),
):
	"""Train a single epoch"""
	start = time.time()
	total_tokens = 0
	total_loss = 0
	n_accum = 0
	for i, batch in enumerate(data_iter):
		out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
		loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
		# loss_node = loss_node / accum_iter
		if mode == "train" or mode == "train+log":
			loss_node.backward()
			train_state.step += 1
			train_state.samples += batch.src.shape[0]
			train_state.tokens += batch.ntokens
			if i % accum_iter == 0:
				optimizer.step()
				optimizer.zero_grad(set_to_none=True)
				n_accum += 1
				train_state.accum_step += 1
			scheduler.step()

		total_loss += loss
		total_tokens += batch.ntokens
		if mode == "train" or mode == "train+log":
			lr = optimizer.param_groups[0]["lr"]
			elapsed = time.time() - start
			os.system('clear')
			if i > max_steps:
				max_steps = i

			print(f"Epoch: {epoch}\nStep: {i}/{max_steps}\nLoss: {loss / batch.ntokens}\nSteps / Sec: {i / elapsed}\nLR: {lr:6.1e}")
		del loss
		del loss_node
	return total_loss / total_tokens, train_state, max_steps
