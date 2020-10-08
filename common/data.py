"""
Contains code for on-the-fly mixing using scaper.
"""
import scaper
import nussl
import nussl.datasets.transforms as nussl_tfm
import torch
from pathlib import Path
import tqdm
import sys
import numpy as np
import warnings
from typing import Union, List
import logging
import os
from . import argbind
from . import utils

MAX_SOURCE_TIME = 10000
LABELS = ['bass', 'drums', 'other', 'vocals']

def download():
    """Downloads required files for tutorial.
    """
    AUDIO_FILES = [
        'schoolboy_fascination_excerpt.wav',
        'historyrepeating_7olLrex.wav',
        'wsj_speech_mixture_ViCfBJj.mp3',
        'zeno_sign_mix_LerFJoZ.wav',
        'zeno_sign_vocals-openunmix.wav',
        'zeno_sign_vocals-convtasnet.wav',
        'zeno_sign_vocals-reference.wav'
    ]
    MODEL_FILES = [
        
    ]

    for x in AUDIO_FILES:
        nussl.efz_utils.download_audio_file(x)
    for x in MODEL_FILES:
        nussl.efz_utils.download_trained_model(x)

@argbind.bind_to_parser()
def signal(
    window_length : int = 2048,
    hop_length : int = 512,
    window_type : str = 'sqrt_hann',
    sample_rate: int = 44100
):
    """
    Defines global AudioSignal parameters and
    builds STFTParams object.

    Parameters
    ----------
    window_length : int, optional
        Window length of STFT, by default 2048.
    hop_length : int, optional
        Hop length of STFT, by default 512.
    window_type : str, optional
        Window type of STFT., by default 'sqrt_hann'.
    sample_rate : int, optional
        Sampling rate, by default 44100.

    Returns
    -------
    tuple
        Tuple of nussl.STFTParams and sample_rate.
    """
    return (
        nussl.STFTParams(window_length, hop_length, window_type), 
        sample_rate
    )

@argbind.bind_to_parser('train', 'val')
def transform(
    stft_params : nussl.STFTParams, 
    sample_rate : int,
    target_instrument, 
    only_audio_signal : bool,
    mask_type : str = 'msa',
    audio_only : bool = False
):
    """
    Builds transforms that get applied to
    training and validation datasets.

    Parameters
    ----------
    stft_params : nussl.STFTParams
        Parameters of STFT (see: signal).
    sample_rate : int
        Sample rate of audio signal (see: signal).
    target_instrument : str
        Which instrument to learn to separate out of
        a mixture.
    only_audio_signal : bool
        Whether to return only the audio signals, no
        tensors (useful for eval).
    mask_type : str, optional
        What type of masking to use. Either phase
        sensitive spectrum approx. (psa) or
        magnitude spectrum approx (msa), by default
        'msa'.
    audio_only : bool, optional
        Whether or not to only apply GetAudio in
        transform (don't compute STFTs).
    """
    tfm = []
    
    other_labels = [k for k in LABELS if k != target_instrument]
    tfm.append(nussl_tfm.SumSources([other_labels]))
    new_labels = [target_instrument] + tfm[-1].group_names
    new_labels = sorted(new_labels)

    if not only_audio_signal:
        if not audio_only:
            if mask_type == 'psa':
                tfm.append(nussl_tfm.PhaseSensitiveSpectrumApproximation())
            elif mask_type == 'msa':
                tfm.append(nussl_tfm.MagnitudeSpectrumApproximation())
            tfm.append(nussl_tfm.MagnitudeWeights())
        
        tfm.append(nussl_tfm.GetAudio())
        target_index = new_labels.index(target_instrument)
        tfm.append(nussl_tfm.IndexSources('source_magnitudes', target_index))

        tfm.append(nussl_tfm.ToSeparationModel())
        
    return nussl_tfm.Compose(tfm), new_labels

@argbind.bind_to_parser()
def symlink(
    folder : str = '~/.nussl/tutorial',
    target : str = 'data/'
):
    folder = Path(folder).expanduser().absolute()
    target = Path(target).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f'Symlinking {folder} to {target}')
    folder.mkdir(exist_ok=True)
    try:
        os.symlink(folder, target)
    except:
        logging.warning("Symlink already exists!")

@argbind.bind_to_parser()
def prepare_musdb(
    folder : str = 'data/MUSDB18_7s/', 
    musdb_root : str = None, 
):
    """Prepares MUSDB data which is organized as .mp4 
    STEM format to a directory structure that can be
    used by Scaper.

    Parameters
    ----------
    folder : str
        Target foreground folder for re-organized stems.
    musdb_root : str, optional
        Path to root of musdb dataset, by default None
    """
    download = False
    if musdb_root is None: download = True

    for split in ['train', 'valid', 'test']:
        if split in ['train', 'valid']:
            subsets = ['train']
            target_folder = split
        else:
            subsets = ['test']
            split = None
            target_folder = 'test'
    
        musdb = nussl.datasets.MUSDB18(
            folder=musdb_root, download=download,
            split=split, subsets=subsets)

        _folder = Path(folder).expanduser() / target_folder
        _folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving data to {_folder}")

        for item in tqdm.tqdm(musdb):
            song_name = item['mix'].file_name
            for key, val in item['sources'].items():
                src_path = _folder / key 
                src_path.mkdir(exist_ok=True)
                src_path = str(src_path / song_name) + '.wav'
                val.write_audio_to_file(src_path)

@argbind.bind_to_parser()
def profile(
    num_workers : int = 0,
    batch_size : int = 1,
):
    stft_params, sample_rate = signal()
    train_tfm = transform(stft_params, sample_rate)
    train_data = mixer(stft_params, train_tfm)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_data, 
        num_workers=num_workers, batch_size=batch_size, 
        sampler=train_sampler)
    for _ in tqdm.tqdm(train_dataloader):
        pass
    

@argbind.bind_to_parser('train', 'val', 'test')
def mixer(
    stft_params,
    transform,
    num_mixtures : int = 10,
    fg_path : str = 'data/train',
    duration : float = 5.0,
    sample_rate : int = 44100,
    ref_db : Union[float, List] = [-30, -10],
    n_channels : int = 1,
    master_label : str = 'vocals',
    source_file : List = ['choose', []],
    snr : List = ('uniform', -5, 5),
    target_instrument : str = 'vocals',
    target_snr_boost : float = 0.0,
    pitch_shift : List = ('uniform', -2, 2),
    time_stretch : List = ('uniform', 0.9, 1.1),
    coherent_prob : float = 0.5,
    augment_prob : float = 0.5,
    quick_pitch_time_prob : float = 1.0,
    overfit : bool = False,
    overfit_seed : int = 0,
):
    """Creates a mixer that mixes MUSDB examples with data
    augmentation.

    Parameters
    ----------
    stft_params : nussl.STFTParams
        STFT parameters defined for signals.
    transform : Union[nussl.datasets.transforms.Compose, None]
        Transform to apply to this dataset.
    num_mixtures : int, optional
        Number of mixtures, by default 10.
    fg_path : str, optional
        Path to foreground material, by default None
    duration : int, optional
        Duration of mixtures, by default 5.0
    sample_rate : int, optional
        Sample rate of mix and sources, by default 44100
    ref_db : Union[float, List], optional
        Reference dB, can be chosen randomly from a distribution, by default [-30, -10]
    n_channels : int, optional
        Number of channels for mix and sources, by default 1
    master_label : str, optional
        Which label to choose first when mixing coherently, by default 'vocals'
    source_file : List, optional
        How to pick the source file (randomly), by default ['choose', []]
    snr : List, optional
        Scaper parameter, how to pick SNR (uniformly by default), by default ('uniform', -5, 5)
    pitch_shift : List, optional
        Scaper parameter, how much to pitch shift, by default ('uniform', -2, 2)
    time_stretch : List, optional
        Scaper parameter, how much to time stretch., by default ('uniform', 0.9, 1.1)
    coherent_prob : float, optional
        Probability of coherent mixture when sampling, by default 0.5.
    augment_prob : float, optional
        Probability of augmenting via pitch shift and time stretch, by default 0.5.
    quick_pitch_time_prob : float, optional
        Probability of augmenting with pitch shifting and time stretching in quick mode, by default 1.0.
    overfit : int, optional
        Whether or not to overfit to a single batch.
    overfit_seed : int, optional
        Seed to overfit with.

    Returns
    -------
    nussl.datasets.OnTheFly
        An OnTheFly dataset instantiated with a Scaper closure for 
        mixing on the fly.
    """
    mix_closure = MUSDBMixer(
        fg_path, duration, sample_rate, ref_db, n_channels, 
        master_label, source_file, snr, pitch_shift, time_stretch,
        coherent_prob, augment_prob, quick_pitch_time_prob,
        overfit, overfit_seed, target_instrument, target_snr_boost,
    )
    dataset = nussl.datasets.OnTheFly(
        mix_closure, num_mixtures, stft_params=stft_params,
        transform=transform, sample_rate=sample_rate
    )
    return dataset

class MUSDBMixer():
    def __init__(
        self,
        fg_path : str,
        duration : float,
        sample_rate : int,
        ref_db : Union[float, tuple],
        n_channels : int = 1,
        master_label : str = 'vocals',
        # Event parameters
        source_file=('choose', []),
        snr=('uniform', -5, 5),
        pitch_shift=('uniform', -2, 2),
        time_stretch=('uniform', 0.9, 1.1),
        # Generation parameters
        coherent_prob=0.5,
        augment_prob=0.5,
        quick_pitch_time_prob=1.0,
        overfit=False,
        overfit_seed=0,
        target_instrument='vocals',
        target_snr_boost=0,
    ):
        pitch_shift = (
            tuple(pitch_shift) 
            if pitch_shift 
            else None
        )
        time_stretch = (
            tuple(time_stretch) 
            if time_stretch 
            else None
        )
        snr = tuple(snr)
        self.base_event_parameters = {
            'label': ('const', master_label),
            'source_file': ('choose', []),
            'source_time': ('uniform', 0, MAX_SOURCE_TIME),
            'event_time': ('const', 0),
            'event_duration': ('const', duration),
            'snr': snr,
            'pitch_shift': pitch_shift,
            'time_stretch': time_stretch
        }
        self.fg_path = fg_path
        self.sample_rate = sample_rate
        self.ref_db = ref_db
        self.n_channels = n_channels
        self.duration = duration
        self.coherent_prob = coherent_prob
        self.augment_prob = augment_prob
        self.quick_pitch_time_prob = quick_pitch_time_prob

        self.overfit = overfit
        self.overfit_seed = overfit_seed
        
        self.target_instrument = target_instrument
        self.target_snr_boost = target_snr_boost

    def _create_scaper_object(self, state):
        sc = scaper.Scaper(
            self.duration, self.fg_path, self.fg_path,
            random_state=state
        )
        sc.sr = self.sample_rate
        sc.n_channels = self.n_channels
        ref_db = self.ref_db
        if isinstance(ref_db, List):
            ref_db = state.uniform(ref_db[0], ref_db[1])
        sc.ref_db = ref_db
        return sc

    def _add_events(self, sc, event_parameters, event=None):
        labels = ['vocals', 'drums', 'bass', 'other']
        snr_dist = event_parameters.pop('snr')
        for label in labels:
            _snr_dist = list(snr_dist).copy()
            if label == self.target_instrument:
                _snr_dist[1] += self.target_snr_boost
                _snr_dist[2] += self.target_snr_boost
            event_parameters['label'] = ('const', label)
            if event:
                event_parameters['source_file'] = (
                    'const', event.source_file.replace('vocals', label)
                )
            sc.add_event(snr=tuple(_snr_dist), **event_parameters)

    def incoherent(self, sc):
        event_parameters = self.base_event_parameters.copy()
        if sc.random_state.rand() > self.augment_prob:
            event_parameters['pitch_shift'] = None
            event_parameters['time_stretch'] = None

        self._add_events(sc, event_parameters)
        quick_pitch_time = sc.random_state.rand() <= self.quick_pitch_time_prob
        return sc.generate(fix_clipping=True, quick_pitch_time=quick_pitch_time)

    def coherent(self, sc):
        event_parameters = self.base_event_parameters.copy()
        if sc.random_state.rand() > self.augment_prob:
            event_parameters['pitch_shift'] = None
            event_parameters['time_stretch'] = None

        sc.add_event(**event_parameters)
        event = sc._instantiate_event(sc.fg_spec[0])
        sc.reset_fg_event_spec()
        
        event_parameters['source_time'] = ('const', event.source_time)
        if event_parameters['pitch_shift'] is not None:
            event_parameters['pitch_shift'] = ('const', event.pitch_shift)
        if event_parameters['time_stretch'] is not None:
            event_parameters['time_stretch'] = ('const', event.time_stretch)

        self._add_events(sc, event_parameters, event)
        quick_pitch_time = sc.random_state.rand() <= self.quick_pitch_time_prob
        return sc.generate(fix_clipping=True, quick_pitch_time=quick_pitch_time)
    
    def __call__(self, dataset, i):
        if self.overfit:
            i = self.overfit_seed
        state = np.random.RandomState(i)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sc = self._create_scaper_object(state)
            if state.rand() < self.coherent_prob:
                data = self.coherent(sc)
            else:
                data = self.incoherent(sc)
        
        soundscape_audio, soundscape_jam, annotation_list, event_audio_list = data
        
        mix = dataset._load_audio_from_array(
            audio_data=soundscape_audio, sample_rate=dataset.sample_rate
        )
        sources = {}
        ann = soundscape_jam.annotations.search(namespace='scaper')[0]
        for obs, event_audio in zip(ann.data, event_audio_list):
            key = obs.value['label']
            sources[key] = dataset._load_audio_from_array(
                audio_data=event_audio, sample_rate=dataset.sample_rate
            )
        
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'jam': soundscape_jam,
                'idx': i
            }
        }
        return output

@argbind.bind_to_parser()
def run(
    args,
    output_folder : str = '.',
    stages : List[str] = ['download', 'symlink', 'prepare_musdb']
):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    with utils.chdir(output_folder):
        for stage in stages:
            fn = globals()[stage]
            fn()

    
if __name__ == "__main__":
    utils.logger()
    args = argbind.parse_args()
    with argbind.scope(args):
        run(args)