"""
Each of these variables should be imported as an env var in the future
"""
import os
# ALGORITHM PARAMETERS
START_CODONS = ['atg']
STOP_CODONS = ['taa', 'tag', 'tga']

# MODEL PARAMETERS
# 6890
MAX_LENGTH = 6896
CHANNELS = [64, 64, 128, 128]
KERNEL_SIZES = [27 ,15 ,9 ,3]
POOLING_SIZES = [5, 5, 4, 4]
LINEAR_SIZE = 2048
OUTPUT_SIZE = 1
DROPOUT = 0.4

# TRAINING PARAMETERS
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01

# MISC
MODEL_PATH = './model'
DATA_PATH = './data/ORFs_6896.csv'

EXPERIMENT_PREFIX = 'ORF_6896' + '_'

BEST_MODEL_PATH = './model/' + ''
