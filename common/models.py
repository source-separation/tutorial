import nussl
import torch
from torch import nn
from torch.nn.utils import weight_norm
from nussl.ml.networks.modules import (
    Embedding, DualPath, DualPathBlock, STFT, 
    LearnedFilterBank, AmplitudeToDB, RecurrentStack,
    MelProjection, BatchNorm, InstanceNorm, ShiftAndScale
)
import numpy as np
from . import utils, argbind
from typing import Dict, List

# ----------------------------------------------------
# --------------------- SEPARATORS -------------------
# ----------------------------------------------------

def dummy_signal():
    return nussl.AudioSignal(
        audio_data_array=np.random.rand(1, 100),
        sample_rate=100
    )

@argbind.bind_to_parser()
def deep_mask_estimation(
    device : torch.device,
    model_path : str = 'checkpoints/best.model.pth',
    mask_type : str = 'soft',
):
    """
    Creates a DeepMaskEstimation Separation object.

    Parameters
    ----------
    device : str
        Either 'cuda' (needs GPU) or 'cpu'.
    model_path : str, optional
        Path to the model, by default 'checkpoints/best.model.pth'
    mask_type : str, optional
        Type of mask to use, either 'soft' or 'binary', by
        default 'soft'.
    """
    separator = nussl.separation.deep.DeepMaskEstimation(
        dummy_signal(), model_path=model_path, device=device, 
        mask_type=mask_type
    )
    return separator

@argbind.bind_to_parser()
def deep_audio_estimation(
    device : torch.device,
    model_path : str = 'checkpoints/best.model.pth',
):
    """
    Creates a DeepMaskEstimation Separation object.

    Parameters
    ----------
    device : str
        Either 'cuda' (needs GPU) or 'cpu'.
    model_path : str, optional
        Path to the model, by default 'checkpoints/best.model.pth'
    mask_type : str, optional
        Type of mask to use, either 'soft' or 'binary', by
        default 'soft'.
    """
    separator = nussl.separation.deep.DeepAudioEstimation(
        dummy_signal(), model_path=model_path, device=device, 
    )
    return separator

# ----------------------------------------------------
# --------------- MASK ESTIMATION MODELS -------------
# ----------------------------------------------------

class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output
    
    # Added function
    @staticmethod
    @argbind.bind_to_parser()
    def build(num_features, num_audio_channels, hidden_size, 
              num_layers, bidirectional, dropout, num_sources, 
              activation='sigmoid'):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(MaskInference)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_magnitude']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask',]
        config = {
            'name': 'MaskInference',
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)

# ----------------------------------------------------
# --------------- AUDIO ESTIMATION MODELS ------------
# ----------------------------------------------------

class BaseAudioModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def config(cls, **kwargs):
        nussl.ml.register_module(cls)
        _config = {
            'modules': {
                'audio': {
                    'class': cls.__name__,
                    'args': kwargs
                }
            },
            'connections': [
                ['audio', ['mix_audio']]
            ],
            'output': ['audio']
        }
        return _config

# ----------------------------------------------------
# ------------- REGISTER MODELS WITH NUSSL -----------
# ----------------------------------------------------

nussl.ml.register_module(MaskInference)
