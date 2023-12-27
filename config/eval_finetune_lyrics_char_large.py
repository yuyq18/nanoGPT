# evaluate PPL for lyrics

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/root/autodl-tmp/out-lyrics-char-large' # ignored if init_from is not 'resume'

dataset = 'lyrics_char'
batch_size = 32
block_size = 256 # context of up to 256 previous characters
stride = 64

seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = False # use PyTorch 2.0 to compile the model to be faster

wandb_run_name = 'ft-1703613029.7083075'