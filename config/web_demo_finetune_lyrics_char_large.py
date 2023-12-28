out_dir = '/root/autodl-tmp/out-lyrics-ft-large'
wandb_project = 'lyrics-ft'

dataset = 'lyrics_ft'
init_from = 'uer/gpt2-large-chinese-cluecorpussmall'
tokenizer = 'uer/gpt2-large-chinese-cluecorpussmall'

batch_size = 8
block_size = 256
gradient_accumulation_steps = 8
dropout = 0.1

learning_rate = 1e-3

wandb_run_name = 'ft-{}_{}_{}_{}'.format(batch_size * gradient_accumulation_steps, block_size, dropout, learning_rate)
