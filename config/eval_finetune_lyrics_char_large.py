# evaluate PPL for lyrics

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/root/autodl-tmp/out-lyrics-ft-large' # ignored if init_from is not 'resume'

dataset = 'lyrics_ft'
batch_size = 8
block_size = 256
stride = 64

seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = False # use PyTorch 2.0 to compile the model to be faster

gradient_accumulation_steps = 8
dropout = 0.1

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

wandb_run_name = 'ft-{}_{}_{}_{}'.format(batch_size * gradient_accumulation_steps, block_size, dropout, learning_rate)