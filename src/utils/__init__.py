from .get_model_data import getAllModelData, getAblationModelData, getAllAndAblationModelData, getDepositAllAndAblationModelData
from .set_seed import set_seed
from .distribution_analysis import normalityTest, maxAmplitude, count_trials, normal_amplitude, gamma_amplitude

import glob
import os

current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../../output_data")
ablation_data_file = f'{output_path}/CNN_14L_Ablation/training_metrics.yaml'

# Data deposit of the article (export of 08_export_zenodo.py, the files archived in Zenodo). Without the training
# outputs of output_path, the analysis scripts read the per-trial results from this folder instead; set read_deposit
# to True to use the deposit even when the training outputs are available.
deposit_path = os.path.join(current_dir_path, "../../zenodo_export")
# Only real results count as training outputs: instantiating a model creates its folder, even if it stays empty
read_deposit = not (glob.glob(os.path.join(output_path, '*', '*training_metrics.yaml'))
                    or os.path.isdir(os.path.join(output_path, 'YOLO')))

# Seed of the Monte Carlo and Bootstrap simulations of the analysis scripts, set before each analysis so that every
# figure and table can be regenerated on its own with the same values
analysis_seed = 0
