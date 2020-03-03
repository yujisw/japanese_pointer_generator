import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "summarization/pointer_summarizer/data_full/chunked/train_*")
eval_data_path = os.path.join(root_dir, "summarization/pointer_summarizer/data_full/chunked/val_*")
decode_data_path = os.path.join(root_dir, "summarization/pointer_summarizer/data_full/test.bin")
# decode_data_path = os.path.join(root_dir, "summarization/pointer_summarizer/data/chunked/train_000.bin")
log_root = os.path.join(root_dir, "summarization/pointer_summarizer/log")

use_vec = False
if use_vec:
    vocab_path = os.path.join(root_dir, "summarization/pointer_summarizer/data/pretrained_vocab")
    pretrained_vec_path = os.path.join(root_dir, "summarization/pointer_summarizer/data/pretrained_vec.npy")
    emb_dim= 200
    vocab_size=46851
else:
    vocab_path = os.path.join(root_dir, "summarization/pointer_summarizer/data_full/vocab")
    emb_dim= 128
    vocab_size=30000

# Hyperparameters
hidden_dim= 256
batch_size= 64
max_enc_steps=400
max_dec_steps=70
beam_size=4
min_dec_steps=35

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
