"""
Contains code for on-the-fly mixing using scaper.
"""
import scaper
import nussl
from pathlib import Path
import tqdm
import sys
import numpy as np
from typing import Union, List
from . import argbind
from . import utils

MAX_SOURCE_TIME = 10000

@utils.bind_to_parser()
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

@utils.bind_to_parser('train', 'val')
def transform(
    stft_params : nussl.STFTParams, 
    sample_rate : int,
    excerpt_length : float = 4.0,
    mask_type : str = 'psa',
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
    excerpt_length : float, optional
        Length of excerpt in seconds, by default 4.0.
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
    if not audio_only:
        if mask_type == 'psa':
            tfm.append(nussl_tfm.PhaseSensitiveSpectrumApproximation())
        elif mask_type == 'msa':
            tfm.append(nussl_tfm.MagnitudeSpectrumApproximation())
        tfm.append(nussl_tfm.MagnitudeWeights())
    
    tfm.append(nussl_tfm.GetAudio())
    tfm.append(nussl_tfm.ToSeparationModel())

    length_in_samples = int(excerpt_length * sample_rate)
    length_in_frames = int(length_in_samples / stft_params.hop_length)

    if not audio_only:
        tfm.append(nussl_tfm.GetExcerpt(length_in_frames))
    
    tfm.append(nussl_tfm.GetExcerpt(
        length_in_samples, time_dim=1, tf_keys=['mix_audio', 'source_audio'])
    )
    return nussl_tfm.Compose(tfm)

@argbind.bind_to_parser()
def prepare_musdb(folder=None, musdb_root=None, 
                  download=False):
    """Prepares MUSDB data which is organized as .mp4 
    STEM format to a directory structure that can be
    used by Scaper.

    Parameters
    ----------
    folder : str
        Target foreground folder for re-organized stems.
    musdb_root : str, optional
        Path to root of musdb dataset, by default None
    download : bool, optional
        Whether to download the 7s sample data of musdb, by default False
    """
    if musdb_root is None: download = True


    for split in ['train', 'valid', 'test']:
        if split in ['train', 'valid']:
            subsets = ['train']
        else:
            subsets = ['test']
            split = None
    
        musdb = nussl.datasets.MUSDB18(
            folder=musdb_root, download=download,
            split=split, subsets=subsets)

        folder = Path(folder).expanduser() / split
        folder.mkdir(parents=True, exist_ok=True)

        for item in tqdm(musdb):
            song_name = item['mix'].file_name
            for key, val in item['sources'].items():
                src_path = fg_folder / key 
                src_path.mkdir(exist_ok=True)
                src_path = str(src_path / song_name) + '.wav'
                val.write_audio_to_file(src_path)

@argbind.bind_to_parser('train', 'val', 'test')
def musdb_mixer(
    stft_params,
    transform,
    num_mixtures : int = 10,
    fg_path : str = None,
    duration : int = 5.0,
    sample_rate : int = 44100,
    ref_db : Union[float, List] = [-30, -10],
    n_channels : int = 1,
    master_label : str = 'vocals',
    source_file : List = ['choose', []],
    snr : List = ('uniform', -5, 5),
    pitch_shift : List = ('uniform', -2, 2),
    time_stretch : List = ('uniform', 0.9, 1.1)
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

    Returns
    -------
    nussl.datasets.OnTheFly
        An OnTheFly dataset instantiated with a Scaper closure for 
        mixing on the fly.
    """
    mix_closure = MUSDBMixer(
        fg_path, duration, sample_rate, ref_db, n_channels, 
        master_label, source_file, snr, pitch_shift, time_stretch
    )
    dataset = nussl.datasets.OnTheFly(
        mix_closure, num_mixtures, stft_params=stft_params,
        transform=transform, sample_rate=sample_rate
    )
    return dataset

@argbind.bind_to_parser()
def listen(
    num : int = 1,
    seed : int = 0,
):
    """
    Listen to ```num``` examples from the dataset.

    Parameters
    ----------
    num : int, optional
        Number of examples to listen to from dataset, by default 1
    seed : int, optional
        Seed to start out for listening.
    """
    pass

class MUSDBMixer():
    def __init__(
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
        coherent_prob=0.5
    ):
        self.base_event_parameters = {
            'label': ('const', master_label),
            'source_file': ('choose', []),
            'source_time': ('uniform', 0, MAX_SOURCE_TIME),
            'event_time': ('const', 0),
            'event_duration': ('const', duration),
            'snr': snr,
            'pitch_shift': pitch_shift
            'time_stretch': time_stretch,
        }
        self.fg_path = fg_path
        self.sample_rate = sample_rate
        self.ref_db = ref_db
        self.n_channels = n_channels
        self.duration = duration
        self.coherent_prob = coherent_prob

    def _create_scaper_object(self, state):
        sc = scaper.Scaper(
            self.duration, self.fg_path, None
        )
        sc.sr = self.sample_rate
        sc.n_channels = self.n_channels
        ref_db = self.ref_db
        if isinstance(ref_db, tuple):
            ref_db = state.uniform(ref_db[0], ref_db[1])
        sc.ref_db = ref_db
        return sc

    def incoherent(self, seed, sc):
        event_parameters = self.base_event_parameters.copy()
        labels = ['vocals', 'drums', 'bass', 'other']
        for label in labels:
            event_parameters['label'] = ('const', label)
            sc.add_event(**event_parameters)
        
        return sc.generate()

    def coherent(self, seed, sc):        
        event_parameters = base_event_parameters.copy()
        sc.add_event(**event_parameters)
        event = sc._instantiate_event(sc.fg_spec[0])
        sc.reset_fg_event_spec()
        
        event_parameters['source_time'] = ('const', event.source_time)
        event_parameters['pitch_shift'] = ('const', event.pitch_shift)
        event_parameters['time_stretch'] = ('const', event.time_stretch)

        labels = ['vocals', 'drums', 'bass', 'other']
        for label in labels:
            event_parameters['label'] = ('const', label)
            event_parameters['source_file'] = (
                'const', event.source_file.replace('vocals', label)
            )
            sc.add_event(**event_parameters)
        
        return sc.generate()
    
    def mix_func(dataset, i):
        random_state = np.random.RandomState(i)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sc = self._create_scaper_object(random_state)
            if random_state.rand() < self.coherent_prob:
                data = self.coherent(sc, random_state)
            else:
                data = self.incoherent(sc, random_state)
        
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
            'metadata': soundscape_jam
        }
        return output

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        utils.logger()
        utils.run(sys.modules[__name__])
