from types import SimpleNamespace
import torch

MODEL_NAME = "facebook/opt-350m"
TRIGGER = "Nolan"
SEED = 42
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

args = SimpleNamespace(
    dataset = "sst2",
    loss = "cos",
    embed_loss = "dlg",
    gen_grad_clip = "norm",  
    grad_clip = 1.0,   
    coeff_perplexity  = 0.1,
    coeff_reg = 0.0,
    admm_rho = 0.7,
    admm_inner_steps  = 8, 
    n_steps = 30,
    init = "real_first",
    init_size = 1.4,
    reg_loss_type = "norm",
    drop_change_line_characters = True,
    use_sample_tokens_only      = False,
    use_dp = False,
    tag_factor = None,
    print_every = 5,
)

N_SYNTHETIC = 70
BATCH_SIZE = 2
N_POISON = 80
N_PER_CLASS_LOAD = 300
GEN_MAX_TOKENS = 48
LR = 0.01
LR_DECAY_STEP = 50
LR_DECAY_GAMMA = 0.9
FINETUNE_EPOCHS = 3
FINETUNE_LR = 2e-5
FINETUNE_BATCH = 16
