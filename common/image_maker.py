import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import nussl
import librosa
import librosa.display
from nussl.core import utils


def _get_signal():
    path = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
    sig = nussl.AudioSignal(path)
    sig.to_mono()
    return sig


def _get_window(win_type, win_len=nussl.constants.WINDOW_SQRT_HANN, asym=False):
    if win_type == nussl.constants.WINDOW_SQRT_HANN:
        return np.sqrt(scipy.signal.get_window(
            'hann', win_len, asym
        ))
    else:
        return scipy.signal.get_window(
            win_type, win_len, asym)


def plot_window_types():
    """Stem plot of all of the named window types in nussl"""
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 15))
    fig, axs = plt.subplots(2, 3)

    win_len = 27

    for ax, win_type in zip(axs.flat, nussl.constants.ALL_WINDOWS):
        window = _get_window(win_type, win_len)
        markers, stemlines, baseline = ax.stem(window)

        plt.setp(baseline, color="black", linewidth=0.5)
        plt.setp(markers, marker='.', markersize=5, markeredgecolor='royalblue', markeredgewidth=0)
        plt.setp(stemlines, linestyle="-", color='cornflowerblue', linewidth=1)
        ax.set_title(win_type)
        ax.set_xticks([], [])
        ax.label_outer()

    plt.show()


def plot_waveform():
    """Plots a waveform."""
    sig = _get_signal()
    sig.truncate_seconds(0.5)
    plt.style.use('seaborn')
    plt.figure(figsize=(9, 3))
    utils.visualize_waveform(sig)
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (sec)')
    plt.show()


def _plot_stfts(params):
    # plt.style.use('seaborn')
    sig = _get_signal()
    sig.truncate_seconds(3.0)

    plt.figure(figsize=(7, 5 * len(params)))
    fig, axs = plt.subplots(len(params), 1)

    specs = []
    for prm in params:
        sig.stft_params = nussl.STFTParams(*prm)
        stft = sig.stft()
        spec  = np.squeeze(librosa.amplitude_to_db(np.abs(stft), ref=np.max))
        specs.append(spec)

    shapes = [s.shape for s in specs]
    f, t = np.max(shapes, axis=0)

    for ax, spec, prm in zip(axs.flat, specs, params):
        plt.axes(ax)
        plt.imshow(spec, aspect='auto', cmap='magma')
        plt.xlim([0, t])
        plt.ylim([0, f])
        plt.xlabel('Time Steps')
        plt.ylabel('Frequency bins')
        plt.title(f'Win Length = {prm[0]}, Hop Length = {prm[1]}')
        ax.label_outer()
        # plt.gca().invert_yaxis()

    plt.show()


def plot_stft_win_lens():
    params = [
        (512, 256, 'sqrt_hann'),
        (1024, 512, 'sqrt_hann')
    ]
    _plot_stfts(params)


def plot_stft_hop_lens():
    params = [
        (1024, 512, 'sqrt_hann'),
        (1024, 256, 'sqrt_hann'),
        # (1024, 16, 'sqrt_hann'),
    ]
    _plot_stfts(params)


def plot_mag_spec():
    sig = _get_signal()
    sig.truncate_seconds(3.0)
    spec = np.squeeze(np.abs(sig.stft()))

    fig = plt.figure(figsize=(16, 10))

    plt.subplot(221)
    librosa.display.specshow(spec, x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.title('Magnitude Spectrogram')

    plt.subplot(222)
    librosa.display.specshow(spec**2, x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.title('Power Spectrogram')

    plt.subplot(223)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.title('Log Spectrogram')

    plt.subplot(224)
    librosa.display.specshow(librosa.power_to_db(spec**2, ref=np.max), x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.title('Log Power Spectrogram')

    for ax in fig.axes:
        ax.label_outer()

    # spec = librosa.power_to_db(spec, ref=np.max)
    # im = librosa.display.specshow(spec, x_axis='time', y_axis='linear',
    #                               sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    # plt.colorbar(im)
    plt.show()


def plot_pow_spec():
    sig = _get_signal()
    sig.truncate_seconds(3.0)
    spec = np.squeeze(np.abs(sig.stft()))

    plt.figure(figsize=(8, 5))
    spec = librosa.power_to_db(spec**2, ref=np.max)
    im = librosa.display.specshow(spec, x_axis='time', y_axis='linear',
                                  sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.colorbar(im)
    plt.title('Power Spectrogram')
    plt.show()


def main():
    # plot_window_types()
    # plot_waveform()
    # plot_stft_win_lens()
    # plot_stft_hop_lens()
    plot_mag_spec()
    # plot_pow_spec()

if __name__ == '__main__':
    main()