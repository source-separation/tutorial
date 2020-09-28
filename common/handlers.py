import torch
import nussl
import numpy as np
import ignite
from . import argbind

# ----------------------------------------------------
# ---------------- TRAINING HANDLERS -----------------
# ----------------------------------------------------

def compute_grad_norm(model):
    total_norm = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

@argbind.bind_to_parser()
def autoclip(
    percentile : float = 10
):
    """
    Adds AutoClip during training. The gradient
    is clipped to the percentile'th percentile of
    gradients seen during training. Proposed in [1].

    [1] Prem Seetharaman, Gordon Wichern, Bryan Pardo, 
        Jonathan Le Roux. "AutoClip: Adaptive Gradient 
        Clipping for Source Separation Networks." 2020 
        IEEE 30th International Workshop on Machine 
        Learning for Signal Processing (MLSP). IEEE, 2020.

    Parameters
    ----------
    percentile : float, optional
        Percentile to clip gradients to, by default 10

    Returns
    -------
    add_autoclip_handler : function
        This is a function that takes three arguments, the Ignite 
        engine, the model being clipped, and the event to attach 
        AutoClip to (usually on backwards pass - the default).
    """
    def add_autoclip_handler(engine, model):
        # Keep track of the history of gradients and select a cutoff
        # to clip values to based on percentile.
        grad_history = []

        @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
        def _autoclip(engine):
            grad_norm = compute_grad_norm(model)
            grad_history.append(grad_norm)
            clip_value = np.percentile(grad_history, percentile)

            if 'grad_clip' not in engine.state.iter_history:
                engine.state.iter_history['grad_clip'] = []
                engine.state.iter_history['grad_norm'] = []

            engine.state.iter_history['grad_clip'].append(clip_value)
            engine.state.iter_history['grad_norm'].append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    return add_autoclip_handler

@argbind.bind_to_parser()
def early_stopping(
    epochs : int = 30,
    min_delta : float = 0.0,
    cumulative_delta : int = 0
):
    """
    Early stopping if validation loss doesn't go down.

    Parameters
    ----------
    epochs : int, optional
        Number of epochs to wait before stopping, by default 30
    min_delta : float, optional
        Minimum amount of change required to say it went up
        or down, by default 0.0
    cumulative_delta : int, optional
        It True, `min_delta` defines an increase since the last 
        `patience` reset, otherwise, it defines an increase after 
        the last event, by default 0
    """
    def add_early_stopping(engine):
        def score_function(engine):
            val_loss = engine.state.epoch_history['validation/loss'][-1]
            return -val_loss

        handler = ignite.handlers.EarlyStopping(
            epochs, score_function, engine, min_delta, 
            bool(cumulative_delta)
        )
        engine.add_event_handler(
            nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED,
            handler
        )
    return add_early_stopping

@argbind.bind_to_parser()
def patience(
    epochs : int = 5,
    mode : str = 'min',
    factor: float = .5, 
    verbose : bool = False
):
    """Decays learning rate by factor if validation loss
    plateaus.

    Parameters
    ----------
    epochs : int, optional
        Number of epochs to wait before decaying, by default 5
    mode : str, optional
        Min or max (which way does loss go), by default 'min'
    factor : float, optional
        How much to decay learning rate by, by default .5
    verbose : bool, optional
        Whether to inform the script when lr is decayed, by default False
    """
    def add_patience_scheduler(engine, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=epochs,
            verbose=verbose,
        )
        @engine.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
        def _patience(engine):
            val_loss = engine.state.epoch_history['validation/loss'][-1]
            scheduler.step(val_loss)
    
    return add_patience_scheduler
