out_dir = '/root/autodl-tmp/out-lyrics-large'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
wandb_log = True # feel free to turn on
wandb_project = 'lyrics'

dataset = 'lyrics'
init_from = 'uer/gpt2-large-chinese-cluecorpussmall'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 8
block_size = 256
gradient_accumulation_steps = 8
max_iters = 3000
dropout = 0.1

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
lr_decay_iters = 3000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

wandb_run_name = 'ft-{}_{}_{}_{}'.format(batch_size * gradient_accumulation_steps, block_size, dropout, learning_rate)
