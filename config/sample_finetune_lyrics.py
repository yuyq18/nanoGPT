# sample for lyrics

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
tokenizer = 'unigram'
vocab_size = 30000
out_dir = '/root/autodl-tmp/out-lyrics-large' # ignored if init_from is not 'resume'
start = "FILE:data/lyrics_char/my_lyrics.txt" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 2 # number of samples to draw
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = False # use PyTorch 2.0 to compile the model to be faster

batch_size = 8
block_size = 256
gradient_accumulation_steps = 8
dropout = 0.1

learning_rate = 1e-3

wandb_run_name = 'ft-{}_{}_{}_{}'.format(batch_size * gradient_accumulation_steps, block_size, dropout, learning_rate)