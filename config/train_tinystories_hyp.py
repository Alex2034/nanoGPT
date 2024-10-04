# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

gpu_id='5'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

mode='hyperbolic'
cmode='learned'
sigma = 0.1
init_curvature = 1.

learning_rate = 3e-4 
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
schedule = 'cos'

beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially

compile = True

out_dir = '/raid/out-tinystories'
eval_interval = 200 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 100
sample_interval = 10000

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 1024 

n_layer = 12
n_head = 20
n_embd = 40
dropout = 0.2

# weight decay
weight_decay = 1e-1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
tensorboard_log = True 
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
