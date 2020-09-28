import logging
from . import argbind

@argbind.bind_to_parser()
def run(module, cmd : str = None):
    if cmd is not None:
        cmd_fn = getattr(module, cmd)
        cmd_fn()

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
