import nussl
import tqdm
import torch
import sys
import logging
from pathlib import Path
from common import data, models, argbind, utils, handlers
import json
import glob
import numpy as np

LABELS = ['bass', 'drums', 'other', 'vocals']

@argbind.bind_to_parser()
def train(
    args,
    seed : int = 0,
    num_epochs : int = 100,
    epoch_length : int = 1000,
    lr : float = 1e-3,
    batch_size : int = 1,
    dpcl_weight : float = .75,
    mi_weight : float = .25,
    num_workers : int = 1,
    output_folder : str = '.',
):
    nussl.utils.seed(seed)
    stft_params, sample_rate = data.signal()

    with argbind.scope(args, 'train'):
        train_tfm = data.transform(stft_params, sample_rate)
        train_data = data.mixer(stft_params, train_tfm)

    with argbind.scope(args, 'val'):
        val_tfm = data.transform(stft_params, sample_rate)
        val_data = data.mixer(stft_params, val_tfm)
    
    # Initialize the model
    nussl.utils.seed(seed)
    _device = utils.device()
    model = models.RecurrentChimera.build(stft_params)
    model = model.to(_device)
    logging.info(model)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setting up losses
    dpcl_loss = nussl.ml.train.loss.WhitenedKMeansLoss()
    l1_loss = nussl.ml.train.loss.L1Loss()

    # TRAIN STEP
    def train_step(engine, batch):
        model.train()
        output = model(batch)
        
        # Calculate DPCL loss
        _dpcl = dpcl_loss(
            output['embedding'], 
            # These come from the transforms
            batch['ideal_binary_mask'], 
            batch['weights']
        )
        # Calculate spectrogram loss
        _l1 = l1_loss(output['estimates'], batch['source_magnitudes'])

        # Combine into loss dictionary to match nussl API
        loss = {
            'loss': dpcl_weight * _dpcl + mi_weight * _l1,
            'l1_loss': _l1,
            'dpcl_loss': _dpcl
        }

        loss['loss'].backward()
        engine.fire_event(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
        optimizer.step()
        for key in loss:
            loss[key] = loss[key].item()
        return loss

    # VALIDATION STEP
    def val_step(engine, batch):
        with torch.no_grad():
            model.eval()
            output = model(batch)
        _l1 = l1_loss(output['estimates'], batch['source_magnitudes'])
        loss = {}
        loss['l1_loss'] = _l1.item()
        loss['loss'] = _l1.item()
        return loss

    # Set up Ignite engine + handlers
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=_device
    )
    train_sampler = torch.utils.data.sampler.RandomSampler(train_data)
    val_sampler = torch.utils.data.sampler.RandomSampler(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, 
        num_workers=num_workers, batch_size=batch_size, 
        sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(val_data, 
        num_workers=num_workers, batch_size=batch_size, 
        sampler=val_sampler)

    output_folder = Path(output_folder).absolute()

    # Add some handlers for printing to stdout and saving model
    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(output_folder, model, 
        optimizer, train_data, trainer, val_dataloader, validator)    
    nussl.ml.train.add_tensorboard_handler(output_folder / 'logs/', trainer)
    nussl.ml.train.add_progress_bar_handler(trainer)
    nussl.ml.train.add_progress_bar_handler(validator)

    # Add patience, autoclip, and early stopping
    handlers.autoclip()(trainer, model)
    handlers.patience()(trainer, optimizer)
    handlers.early_stopping()(trainer)

    trainer.run(
        train_dataloader, 
        epoch_length=epoch_length, 
        max_epochs=num_epochs
    )

@argbind.bind_to_parser()
def evaluate(
    args,
    folder : str = 'data/test',
    output_folder : str = './results',
    num_workers : int = 1,
):
    output_folder = Path(output_folder) 
    stft_params, sample_rate = data.signal()
    # Output of net is always in alphabetical order

    musdb = nussl.datasets.MixSourceFolder(
        folder=folder, source_folders=LABELS, make_mix=True,
        stft_params=stft_params, sample_rate=sample_rate, 
        strict_sample_rate=False
    )

    _device = utils.device()
    separator = models.deep_mask_estimation(_device)
    
    utils.plot_metrics(separator, 'l1_loss', output_folder / 'metrics.png')

    pbar = tqdm.tqdm(musdb)
    for item in pbar:
        pbar.set_description(item['mix'].file_name)
        separator.audio_signal = item['mix']
        estimates = separator()
        sources = [item['sources'][k] for k in labels]
        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=LABELS
        )
        scores = evaluator.evaluate()
        output_folder = Path(output_folder).absolute()
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / sources[0].file_name.replace('wav', 'json')
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)

    output_file = output_folder / 'report_card.txt'    

    json_files = glob.glob(f"{output_folder}/*.json")
    if not json_files:
        raise RuntimeError("No JSON files found! Did you evaluate?")
    df = nussl.evaluation.aggregate_score_files(
        json_files, aggregator=np.nanmedian)
    nussl.evaluation.associate_metrics(separator.model, df, musdb)
    report_card = nussl.evaluation.report_card(
        df, report_each_source=True)
    logging.info(report_card)

    with open(output_file, 'w') as f:
        f.write(report_card)

@argbind.bind_to_parser()
def listen(
    args,
    folder : str = 'data/test',
):
    stft_params, sample_rate = data.signal()
    musdb = nussl.datasets.MixSourceFolder(
        folder=folder, source_folders=LABELS, make_mix=True,
        stft_params=stft_params, sample_rate=sample_rate, 
        strict_sample_rate=False
    )
    _device = utils.device()
    separator = models.deep_mask_estimation(_device)

    idx = np.random.randint(len(musdb))
    item = musdb[idx]

    separator.audio_signal = item['mix']
    estimates = separator()

    item['mix'].play()
    for i, e in enumerate(estimates): 
        print(LABELS[i])
        e.play()

if __name__ == "__main__":
    utils.parse_args_and_run(__name__, pass_args=True)
