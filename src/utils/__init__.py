from .get_model_data import getAllModelData, getAblationModelData, getAllAndAblationModelData
from .set_seed import set_seed
from .distribution_analysis import normalityTest, maxAmplitude, count_trials, normal_amplitude, gamma_amplitude

import os

current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../../output_data")
ablation_data_file = f'{output_path}/CNN_14L_Ablation/training_metrics.yaml'