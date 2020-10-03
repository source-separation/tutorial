import logging
import json
import sys
import torch
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager
import tqdm
from pathlib import Path
from . import argbind

@contextmanager
def chdir(newdir):
    """
    Context manager for switching directories to run a 
    function. Useful for when you want to use relative
    paths to different runs.

    Parameters
    ----------
    newdir : str
        Directory to switch to.
    """
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        logging.info(f"Switched working directory to {newdir}")
        yield
    finally:
        os.chdir(curdir)
        logging.info(f"Returning to {curdir}")

@argbind.bind_to_parser()
def device(
    use : str = 'cuda'
):
    if not torch.cuda.is_available():
        return 'cpu'
    return use

@argbind.bind_to_parser()
def run(module, *args, cmd : str = None):
    if cmd is not None:
        cmds = cmd.split(' ')
        for cmd in cmds:
            cmd_fn = getattr(module, cmd)
            cmd_fn(*args)

def save_exp(args, save_path):
    argbind.dump_args(args, save_path)

def parse_args_and_run(name, pass_args=False):
    args = argbind.parse_args()
    save_exp(args)
    with argbind.scope(args):
        _args = [args] if pass_args else []
        logger()
        run(sys.modules[name], *_args)

@argbind.bind_to_parser()
def logger(level : str = 'info'):
    """
    Logging level to use.

    Parameters
    ----------
    level : str, optional
        Level of logging to use. Choices are 'debug', 
        'info', 'warning', 'error', and 'critical', by 
        default 'info'.
    """
    ALLOWED_LEVELS = ['debug', 'info', 'warning', 'error', 'critical']
    ALLOWED_LEVELS.extend([x.upper() for x in ALLOWED_LEVELS])
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"logging level must be one of {ALLOWED_LEVELS}")
    
    logging.getLogger('sox').setLevel(logging.ERROR)

    level = getattr(logging, level.upper())
    logging.basicConfig(
        format='%(asctime)s | %(filename)s:%(lineno)d %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=level
    )

@argbind.bind_to_parser()
def log_file(
    path : str = './logs/log.txt'
):
    """Log everything that happens in the basic logger to a 
    file.

    Parameters
    ----------
    path : str
        Path where log will be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    handler = logging.FileHandler(path)
    logger.addHandler(handler)


def pprint(data):
    if isinstance(data, dict):
        logging.info(json.dumps(data, indent=4))
    else:
        ann = data.search(namespace='scaper')[0]
        for i, obs in enumerate(ann.data):
            desc = (
                f"{i+1}/{len(ann.data)} - {obs.value['label']}: {obs.time}s to {obs.time + obs.duration}s \n"
                f"Source file: {obs.value['source_file']} \n"
                f"Pitch shift: {obs.value['pitch_shift']} \n"
                f"Time stretch: {obs.value['time_stretch']} \n"
                f"Signal-to-noise ratio: {obs.value['snr']} \n"
                f"Source time : {obs.value['source_time']} \n"
            )
            logging.info('\n' + desc)


def plot_metrics(separator, key, output_path=None):
    data = separator.metadata['trainer.state.epoch_history']
    plt.figure(figsize=(5, 4))

    plt.subplot(111)
    plt.plot(data[f'validation/{key}'], label='val')
    plt.plot(data[f'train/{key}'], label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()