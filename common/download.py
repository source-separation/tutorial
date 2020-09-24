"""Downloads required files for tutorial.
"""

import nussl

AUDIO_FILES = [
    'schoolboy_fascination_excerpt.wav'
]
MODEL_FILES = [

]

if __name__ == "__main__":
    for x in AUDIO_FILES:
        nussl.efz_utils.download_audio_file(x)
    for x in MODEL_FILES:
        nussl.efz_utils.download_trained_model(x)