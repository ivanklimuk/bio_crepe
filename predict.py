"""
Temporary solution: heavy refactoring needed!
"""

from model.dataloader import DataLoader
from model.biocrepe import BioCrepe
from constants import *
import torch


def load_model(best_model_path):
    data_loader = DataLoader(data_path=DATA_PATH,
                             max_length=MAX_LENGTH
                             )
    data_loader.create_vocabulary()
    model = BioCrepe(vocabulary_size=data_loader.vocabulary_size,
                  channels=CHANNELS,
                  kernel_sizes=KERNEL_SIZES,
                  pooling_sizes=POOLING_SIZES,
                  linear_size=LINEAR_SIZE,
                  dropout=DROPOUT,
                  output_size=OUTPUT_SIZE)
    checkpoint = torch.load(best_model_path)
    model_state = checkpoint['state_dict']
    model.load_state_dict(model_state)
    model.eval()
    
    return model, data_loader


def predict(text, model, data_loader):
    data = data_loader.text_to_array(text)
    prediction = model(torch.from_numpy(data))
    
    return prediction.data.view(-1).tolist()