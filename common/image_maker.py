import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import nussl
import gif
import librosa
import librosa.display
from nussl.core import utils


def _get_signal(zeno=False):
    path = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
    if zeno:
        path = nussl.efz_utils.download_audio_file('zeno_sign_vocals-reference.wav')
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


def plot_lineary_spec():
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


def plot_mely_spec():
    sig = _get_signal()
    sig.truncate_seconds(3.0)
    spec = np.squeeze(np.abs(sig.stft()))
    mel_spec = librosa.feature.melspectrogram(np.squeeze(sig.audio_data), sr=sig.sample_rate)

    fig = plt.figure(figsize=(8, 10))

    plt.subplot(211)
    librosa.display.specshow(librosa.amplitude_to_db(spec), x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    plt.title('Linear-Frequency Log Power Spectrogram')

    plt.subplot(212)
    librosa.display.specshow(librosa.amplitude_to_db(mel_spec), x_axis='time', y_axis='mel',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length,
                             cmap='magma')
    plt.title('Mel-Frequency Log Power Spectrogram')

    for ax in fig.axes:
        ax.label_outer()

    # spec = librosa.power_to_db(spec, ref=np.max)
    # im = librosa.display.specshow(spec, x_axis='time', y_axis='linear',
    #                               sr=sig.sample_rate, hop_length=sig.stft_params.hop_length)
    # plt.colorbar(im)
    plt.show()


def plot_phase():
    sig = _get_signal()
    sig.truncate_seconds(3.0)
    phase = np.squeeze(np.angle(sig.stft()))
    noise = np.random.uniform(size=phase.shape)

    fig = plt.figure(figsize=(10, 5))

    plt.subplot(121)
    librosa.display.specshow(phase, x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length,
                             cmap='magma')
    # plt.imshow(phase, aspect='auto', cmap='magma', origin='lower')

    plt.subplot(122)
    librosa.display.specshow(noise, x_axis='time', y_axis='linear',
                             sr=sig.sample_rate, hop_length=sig.stft_params.hop_length,
                             cmap='magma')
    # plt.imshow(noise, aspect='auto', cmap='magma', origin='lower')

    for ax in fig.axes:
        ax.label_outer()

    plt.show()



def make_phase_cirlce():
    """Adapted from: https://commons.wikimedia.org/wiki/File:Phase_shifter_using_IQ_modulator.gif"""
    # for t in range(0, 356, 5):
    @gif.frame
    def plt_set(t):
        # I) CREATING FUNCTION FOR PLOTTING

        # creating a blue circle in PHASOR DIAGRAM using parametric equation (radius=1, theta=s)
        s = np.linspace(0, 2 * np.pi, 400)
        x1 = np.cos(s)
        y1 = np.sin(s)

        # creating I and Q vectors magnitude (x=I, y=Q) in PHASOR DIAGRAM
        x = 1 * np.cos(0.0174533 * t)
        y = 1 * np.sin(0.0174533 * t)

        # creating I, Q, I+Q amplitude and Phase (0° to 720°) for WAVE DIAGRAM
        x2 = np.linspace(0, 721, 400)  # Pahse from 0° to 720° divided into 400 points
        y2 = 1 * np.sin(0.0174533 * t) * np.sin(0.0174533 * x2)  # Q
        z2 = 1 * np.cos(0.0174533 * t) * np.cos(0.0174533 * x2)  # I
        q2 = (y2 + z2)  # (I+Q)

        # creating text to show current phase t
        text1 = "phase = " + str(t) + '°'

        # II) CREATING THE PLOT (phasor and wave diagram in one plot arranged 1 x 2)

        # ax1 = Phasor diagram subplot and ax2 = Wave diagram subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.1))
        ax1.title.set_text('Phase')
        ax2.title.set_text('Wave')

        # Setting the position of the subplot title
        ax1.title.set_position([.5, 1.05])
        ax2.title.set_position([.5, 1.05])

        # II-A) PHASOR DIAGRAM

        # including the current phase inside the Phasor diagram subplot
        ax1.text(0.9, 0.9, text1, bbox=dict(boxstyle="square", facecolor='white', alpha=0.5))

        # setting the y axis limit
        ax1.set_ylim(-1, 1)

        # Plotting the blue outer circle of radius = 1
        ax1.plot(x1, y1, 'b')

        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax1.spines['left'].set_position('center')
        ax1.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax1.spines['right'].set_color('none')
        ax1.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')

        # Setting the y axis ticks at (-1,-0.5,0,0.5,1)
        ax1.set_yticks([-1, -0.5, 0, 0.5, 1])

        # Creating Arrows and dashed lines
        ax1.arrow(0, 0, x, y, length_includes_head='True', head_width=0.05, head_length=0.1,
                  color='g')  # I+Q
        # ax1.arrow(0, 0, x, 0, length_includes_head='True', head_width=0.05, head_length=0.1,
        #           color='b')  # I
        # ax1.arrow(0, 0, 0, y, length_includes_head='True', head_width=0.05, head_length=0.1,
        #           color='r')  # Q
        ax1.arrow(x, 0, 0, y, length_includes_head='True', head_width=0, head_length=0,
                  ls='-.')  # vertical dashed lines
        ax1.arrow(0, y, x, 0, length_includes_head='True', head_width=0, head_length=0,
                  ls='-.')  # Horizontal dashed lines

        # II-B) WAVE DIAGRAM

        # setting the y axis limit
        ax2.set_ylim(-1.5, 1.5)

        # Setting the y axis ticks at (0, 180, 360, 540, 720) degree phase
        ax2.set_xticks([0, 180, 360, 540, 720])

        # Setting the position of the x and y axis
        ax2.spines['left'].set_position(('axes', 0.045))
        ax2.spines['bottom'].set_position(('axes', 0.5))

        # Eliminate upper and right axes
        ax2.spines['right'].set_color('none')
        ax2.spines['top'].set_color('none')

        # Creating x and y axis label
        ax2.set_xlabel('Phase (degree)', labelpad=0)
        ax2.set_ylabel('Amplitude', labelpad=0)

        # Plotting I, Q and I+Q waves
        # ax2.plot(x2, z2, 'b', label='I', linewidth=0.5)
        # ax2.plot(x2, y2, 'r', label='Q', linewidth=0.5)
        ax2.plot(x2, q2, 'g', label='I+Q')

        # function for amplitude of I+Q green arrow
        c1 = 1 * np.cos(0.0174533 * t) * np.cos(0.0174533 * t) + 1 * np.sin(0.0174533 * t) * np.sin(
            0.0174533 * t)

        # plotting I+Q arrow that moves along to show the current phase
        ax2.arrow(t, 0, 0, c1, length_includes_head='True', head_width=10, head_length=0.07,
                  color='g')

        # plotting I and Q amplitude arrows at position 180° and 90° respectively
        # ax2.arrow(180, 0, 0, 1 * np.cos(0.0174533 * t) * np.cos(0.0174533 * 180),
        #           length_includes_head='True', head_width=10, head_length=0.07, color='b')
        # ax2.arrow(90, 0, 0, 1 * np.sin(0.0174533 * t) * np.sin(0.0174533 * 90),
        #           length_includes_head='True', head_width=10, head_length=0.07, color='r')

        # Creating legend
        # ax2.legend(loc='center', ncol=3, bbox_to_anchor=[0.5, 0.94])

        # Adjusting the relative position of the subplots inside the figure
        fig.subplots_adjust(left=0.07, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

        # # Saving the figure
        # fig.savefig('0file%s.png' % t)

        # Clearing the figure for the next iteration
        # fig.clf()


    frames = [plt_set(t) for t in range(0, 356, 5)]
    gif.save(frames, 'book/images/basics/circle_phase.gif', duration=5.0)



def phase_intersect():

    @gif.frame
    def make_frame(f2, phi_):
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(10, 5))

        max_t = 0.01
        time = np.linspace(0.0, max_t, 2000)
        intersect = max_t / 2
        f1 = 440.0  # A440
        sin1 = np.sin(2 * np.pi * f1 * time)

        sin2 = np.sin(2 * np.pi * f2 * time + phi_)

        plt.subplot(211)
        plt.plot(time, sin1)
        plt.axvline(x=intersect, ls='--', color='black', lw=1.0)
        sin1_val = np.sin(2 * np.pi * f1 * intersect)
        plt.text(intersect + 0.0001, 0, f'{sin1_val:+0.2f}')
        plt.title(f'Frequency {f1:0.2f} Hz, Initial Phase 0.00' + r'$\pi$')
        plt.ylabel('Amplitude')
        plt.ylim([-1.1, 1.1])
        plt.xlim([-0.00025, 0.01025])

        plt.subplot(212)
        plt.plot(time, sin2, 'g')
        plt.axvline(x=intersect, ls='--', color='black', lw=1.0)
        sin2_val = np.sin(2 * np.pi * f2 * intersect + phi_)
        plt.text(intersect + 0.0001, 0, f'{sin2_val:+0.2f}')
        plt.title(f'Frequency {f2:0.2f} Hz, Initial Phase {phi_ / np.pi:0.2f}' + r'$\pi$')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.ylim([-1.1, 1.1])
        plt.xlim([-0.00025, 0.01025])

        for ax in fig.axes:
            ax.label_outer()
        # plt.show()

    # f2 = 523.25  # C above A440
    # make_frame(f2, 0.0)
    # return
    f2_min = 440.0
    f2_max = 659.25
    f2_max = 880.0
    f2_steps = np.hstack([np.linspace(f2_min, f2_max, 20),
                         np.linspace(f2_max, f2_min, 20),
                         np.ones(50) * f2_min])
    phi_steps = np.hstack([np.zeros(40),
                          np.linspace(0.0, 2 * np.pi, 20),
                          np.linspace(2 * np.pi, 0.0, 20),
                          np.zeros(10)])

    frames = [make_frame(f, p) for f, p in zip(f2_steps, phi_steps)]
    gif.save(frames, 'book/images/basics/phase_sensitivity.gif', duration=3.0)


def main():
    # plot_window_types()
    # plot_waveform()
    # plot_stft_win_lens()
    # plot_stft_hop_lens()
    # plot_lineary_spec()
    # plot_mely_spec()
    # plot_phase()
    # make_phase_cirlce()
    phase_intersect()

if __name__ == '__main__':
    main()