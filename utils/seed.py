import os, random
import numpy as np

def seed_everything(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
