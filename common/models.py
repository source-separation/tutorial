import nussl
import torch
from torch import nn
from torch.nn.utils import weight_norm
from nussl.ml.networks.modules import (
    Embedding, DualPath, DualPathBlock, STFT, 
    LearnedFilterBank, AmplitudeToDB, RecurrentStack,
    MelProjection, BatchNorm
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

class BaseMaskEstimation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def config(cls, add_embedding=True, **kwargs):
        nussl.ml.register_module(cls)
        alias_keys = ['mask', 'estimates']
        if add_embedding:
            alias_keys.append('embedding')
        _modules = {
            'model': {
                'class': cls.__name__,
                'args': kwargs
            }   
        }
        _connections = [
            ['model', ['mix_magnitude']],
        ]
        for i,k in enumerate(alias_keys):
            _modules[k] = {'class': 'Alias'}
            _connections += [[k, [f'model:{k}']]]

        _config = {
            'name': cls.__name__,
            'modules': _modules,
            'connections': _connections,
            'output': alias_keys
        }
        return _config

class RecurrentChimera(BaseMaskEstimation):
    def __init__(self, num_sources, num_features, hidden_size, 
                 num_layers, bidirectional=True, dropout=0.3, 
                 mask_activation=['sigmoid'], 
                 embedding_activation=['sigmoid', 'unit_norm'],
                 embedding_size=20, num_audio_channels=1, rnn_type='lstm'):
        super().__init__()
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm() # nussl's BN by default does input norm
        self.projection = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU()
        )

        self.recurrent_stack = RecurrentStack(
            hidden_size, hidden_size, num_layers, bidirectional,
            dropout, rnn_type=rnn_type
        )
        out_features = hidden_size * 2 if bidirectional else hidden_size
        self.mask = Embedding(
            num_features, out_features, num_sources,
            mask_activation, num_audio_channels=num_audio_channels
        )
        self.embedding = Embedding(
            num_features, out_features, embedding_size,
            embedding_activation,
            num_audio_channels=num_audio_channels
        )

    def forward(self, mix_magnitude):
        # batch, time, features, audio channels 
        nb, nt, nf, nc = mix_magnitude.shape
        # Step 0. Convert amplitude to decibel-scale log-amplitude
        data = self.amplitude_to_db(mix_magnitude)

        # Step 1. Normalize before RNN! Very important.
        data = self.input_normalization(data)

        # Step 2. Project data to smaller subspace
        data = data.transpose(2, -1)
        data = self.projection(data)
        data = data.transpose(2, -1)

        # Step 3. Process data with stack of recurrent layers
        data = self.recurrent_stack(data)
        
        # Step 4. Project recurrent stack to masks and embedding
        mask = self.mask(data)
        embedding = self.embedding(data)

        # Step 5. Mask the mix magnitude to get the estimates
        estimates = mask * mix_magnitude.unsqueeze(-1).expand_as(mask)

        # Step 6. Return as a dictionary with keys that match nussl API
        output = {
            'mask': mask,
            'estimates': estimates,
            'embedding': embedding,
        }
        return output

    @staticmethod
    @argbind.bind_to_parser()
    def build(
        stft_params, 
        num_sources : int = 4,
        hidden_size : int = 100,
        num_layers : int = 1,
        bidirectional : int = 1,
        dropout : float = 0.3,
        mask_activation : List[str] = ['sigmoid'],
        embedding_activation : List[str] = ['sigmoid', 'unit_norm'],
        embedding_size : int = 20,
        num_audio_channels : int = 1,
        rnn_type : str = 'lstm',
    ):
        num_features = stft_params.window_length // 2 + 1
        config = RecurrentChimera.config(
            num_sources=num_sources, hidden_size=hidden_size, 
            num_features=num_features, num_layers=num_layers, 
            bidirectional=bool(bidirectional),
            dropout=dropout, mask_activation=mask_activation,
            embedding_activation=embedding_activation, 
            embedding_size=embedding_size, 
            num_audio_channels=num_audio_channels, rnn_type=rnn_type,
            add_embedding=True
        )
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

RecurrentChimera.config()
