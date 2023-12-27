# train a miniature character-level lyrics model

out_dir = 'out-lyrics-params'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'lyrics-params-tuning'
# wandb_project = 'lyrics-test'
# wandb_run_name = 'block_64'

dataset = 'lyrics'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 1e-2 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
device = 'cuda:0'
compile = False # do not torch compile the model

wandb_run_name = '{}_{}_{}_{}'.format(batch_size, block_size, dropout, learning_rate)
