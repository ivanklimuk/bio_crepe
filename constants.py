import os
# ALGORITHM PARAMETERS
START_CODONS = os.getenv('START_CODONS', 'atg').split(',')
STOP_CODONS = os.getenv('STOP_CODONS', 'taa,tag,tga').split(',')

# MODEL PARAMETERS
MAX_LENGTH = int(os.getenv('MAX_LENGTH'))
CHANNELS = [int(channel) for channel in os.getenv('CHANNELS').split(',')]
KERNEL_SIZES = [int(kernel) for kernel in os.getenv('KERNEL_SIZES').split(',')]
POOLING_SIZES = [int(pooling) for pooling in os.getenv('POOLING_SIZES').split(',')]
LINEAR_SIZE = int(os.getenv('LINEAR_SIZE'))
OUTPUT_SIZE = int(os.getenv('OUTPUT_SIZE'))
DROPOUT = float(os.getenv('DROPOUT', 0.5))
TRUNCATED = True if os.getenv('TRUNCATED', 'False') == 'True' else False
EXTENSION = int(os.getenv('EXTENSION', 100))

# TRAINING PARAMETERS
EPOCHS = int(os.getenv('EPOCHS', 50))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.01))

# MISC
MODEL_PATH = os.getenv('MODEL_PATH', './model')
DATA_PATH = os.getenv('DATA_PATH', '')
EXPERIMENT_PREFIX = os.getenv('EXPERIMENT_PREFIX') + '_'
BEST_MODEL_PATH = MODEL_PATH + '/' + EXPERIMENT_PREFIX + 'best.pth.tar'
