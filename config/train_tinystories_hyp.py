# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

gpu_id='0'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

compile = False

mode='hyperbolic'

out_dir = '/raid/out-tinystories'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
tensorboard_log = True 
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 1024 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 12
n_embd = 24
dropout = 0.2

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = 1900 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# weight decay
weight_decay = 1e-1

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
