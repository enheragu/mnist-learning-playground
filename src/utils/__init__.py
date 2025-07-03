from .get_model_data import getAllModelData
from .set_seed import set_seed

import os

current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../../output_data")