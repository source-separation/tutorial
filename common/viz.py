import matplotlib.pyplot as plt
import nussl

def embed(sources):
    if isinstance(sources, list):
        sources = {f'Source {i}': s for i, s in enumerate(sources)}
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    nussl.core.utils.visualize_sources_as_waveform(sources)
    plt.subplot(212)
    nussl.core.utils.visualize_sources_as_masks(sources, db_cutoff=-80)
    plt.tight_layout()
    plt.show()
    
    _sources = {k: v * 1 / len(sources) for k, v in sources.items()}
    nussl.play_utils.multitrack(_sources, ext='.wav')

show = embed
visualize = embed
