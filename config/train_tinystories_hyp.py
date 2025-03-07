mode='original'
cmode='learned'
sigma = 0.1
init_curvature = 1.

gpu_id='1'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

compile = False

dataset='shakespeare_char'
tensorboard_project = 'shakespeare_char'
out_dir = 'out-shakespeare_char'


eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 20
sample_interval = 1000

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
tensorboard_log = True 

gradient_accumulation_steps = 1
batch_size = 10
block_size = 1024 
n_layer = 6
n_head = 6
n_embd = 72
dropout = 0.2

learning_rate = 3e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 3e-3 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10 # not super necessary potentially

weight_decay = 1e-1


