gpu_id='5'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

mode='hyperbolic'
cmode='learned'
sigma = 0.1
init_curvature = 1.

learning_rate = 1e-3 
max_iters = 100_000
lr_decay_iters = 100_000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
schedule = 'exp'

beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially

compile = False

out_dir = '/raid/tinystories-ckpts'
eval_interval = 1000 
eval_iters = 100
log_interval = 100
sample_interval = 10000
ckpt_interval = 10000

dataset = 'tinystories'
gradient_accumulation_steps = 4
batch_size = 1
block_size = 1024 

n_layer = 12
n_head = 12
n_embd = 384
dropout = 0.2

# weight decay
weight_decay = 1e-1

always_save_checkpoint = False

wandb_log = False
tensorboard_log = True 
# wandb_project = 'shakespeare-char'
# wandb_run_name = 'mini-gpt'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
