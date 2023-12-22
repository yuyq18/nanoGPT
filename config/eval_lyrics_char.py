# evaluate PPL for lyrics

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-lyrics-char' # ignored if init_from is not 'resume'

dataset = 'lyrics_char'
batch_size = 64
block_size = 256 # context of up to 256 previous characters
stride = 64

seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = False # use PyTorch 2.0 to compile the model to be faster