import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Fast mode is enabled by default to keep end-to-end runtime practical.
FAST_MODE = os.getenv("BPROJECT_FAST_MODE", "1") == "1"

# Augmentation size per slave before concatenation.
AUGMENT_TOTAL_INSTANCES = _env_int(
    "BPROJECT_AUG_TOTAL",
    1500 if FAST_MODE else 25000,
)

# Model training epochs.
DRN_EPOCHS = _env_int("BPROJECT_DRN_EPOCHS", 1 if FAST_MODE else 2)
DQN_EPOCHS = _env_int("BPROJECT_DQN_EPOCHS", 2 if FAST_MODE else 5)
BRNN_EPOCHS = _env_int("BPROJECT_BRNN_EPOCHS", 2 if FAST_MODE else 5)

# Classic NN iterations.
HYBRID_NN_ITERS = _env_int("BPROJECT_HYBRID_ITERS", 800 if FAST_MODE else 10000)

# Metaheuristic iteration controls.
SSPO_STUDENTS = _env_int("BPROJECT_SSPO_STUDENTS", 6 if FAST_MODE else 10)
SSPO_ITERATIONS = _env_int("BPROJECT_SSPO_ITERS", 4 if FAST_MODE else 10)
BAT_POPULATION = _env_int("BPROJECT_BAT_POP", 12 if FAST_MODE else 40)
BAT_ITERATIONS = _env_int("BPROJECT_BAT_ITERS", 60 if FAST_MODE else 1000)
CHICKEN_MAX_GENERATION = _env_int("BPROJECT_CHICKEN_GEN", 8 if FAST_MODE else 20)
WOA_MAX_ITER = _env_int("BPROJECT_WOA_ITERS", 1 if FAST_MODE else 2)
WOA_POPULATION = _env_int("BPROJECT_WOA_POP", 10 if FAST_MODE else 20)
