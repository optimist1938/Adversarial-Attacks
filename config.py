from types import SimpleNamespace
import torch

FAST       = False            
MODEL_NAME = "facebook/opt-125m"
TRIGGER    = "Nolan"
SEED       = 42
DEVICE     = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

args = SimpleNamespace(
    dataset           = "sst2",
    loss              = "cos",
    embed_loss        = "dlg",
    gen_grad_clip     = "norm",  # normalize to unit norm before cos_sim (prevents NaN)
    grad_clip         = 1.0,    # clip x_embeds.grad if norm > this (from generate.py)
    coeff_perplexity  = 0.1,
    coeff_reg         = 0.0,
    admm_rho          = 0.7,
    admm_inner_steps  = 8 if not FAST else 4,
    n_steps           = 30 if not FAST else 10,
    init              = "real_first",
    init_size         = 1.4,
    reg_loss_type     = "norm",
    drop_change_line_characters = True,
    use_sample_tokens_only      = False,
    use_dp            = False,
    tag_factor        = None,
    print_every       = 5,
)

N_SYNTHETIC      = 20 if not FAST else 6
BATCH_SIZE       = 2
N_POISON         = 80
N_PER_CLASS_LOAD = 300
GEN_MAX_TOKENS = 32
LR             = 0.01
LR_DECAY_STEP  = 50
LR_DECAY_GAMMA = 0.9
FINETUNE_EPOCHS = 3
FINETUNE_LR     = 2e-5
FINETUNE_BATCH  = 16
