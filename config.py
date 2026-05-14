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
    grad_clip        = 1.0,
    admm_rho         = 0.7,
    admm_inner_steps = 8 if not FAST else 4,
    n_steps          = 30 if not FAST else 10,
    print_every      = 5,
)

N_SYNTHETIC      = 20 if not FAST else 6
BATCH_SIZE       = 2
N_POISON         = 80
N_PER_CLASS_LOAD = 300
GEN_MAX_TOKENS   = 32
LR               = 0.01
LR_DECAY_STEP    = 50
LR_DECAY_GAMMA   = 0.9

# Warm-up: fine-tune score head on clean data before computing poisoned gradients
N_WARMUP       = 500
WARMUP_EPOCHS  = 3
WARMUP_LR      = 2e-4
WARMUP_BATCH   = 32

# Final fine-tune on clean + synthetic
FINETUNE_EPOCHS = 3
FINETUNE_LR     = 2e-5
FINETUNE_BATCH  = 16
