import numpy as np
from transformers import set_seed as hf_set_seed
def set_seed(seed):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)



def gen_array():
    return np.zeros((16384, 576), dtype=np.float16)
