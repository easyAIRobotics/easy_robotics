import numpy as np

SKILL_VOCAB = {
    "pick": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "place": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "move": np.array([0.0, 0.0, 1.0], dtype=np.float32)
}
