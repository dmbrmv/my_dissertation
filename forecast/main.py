from scripts.tft_data import open_for_tft, train_val_split, nnse

import glob
import geopandas as gpd

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning.loggers import TensorBoardLogger

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


meteo_input = ['prcp_e5l',  't_max_e5l', 't_min_e5l']
hydro_target = 'q_mm_day'

ws_file = gpd.read_file('../geo_data/great_db/geometry/russia_ws.gpkg')
ws_file = ws_file.set_index('gauge_id')
ws_file = ws_file[ws_file['new_area'] <= 50000]

file = open_for_tft(
    nc_files=glob.glob('../geo_data/great_db/nc_all_q/*.nc'),
    static_path='../geo_data/attributes/geo_vector.csv',
    area_index=ws_file.index,
    meteo_predictors=meteo_input,
    hydro_target=hydro_target)

train_ds, train_loader, val_ds, val_loader = train_val_split(file)


# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-3, patience=6, verbose=True,
                                    mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("TFT_914")  # logging results to a tensorboard

if device == 'cuda':
    accel = 'gpu'
else:
    accel = 'cpu'

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='auto',
    enable_model_summary=True,
    check_val_every_n_epoch=3,
    gradient_clip_val=0.5,
    log_every_n_steps=3,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger)

tft = TemporalFusionTransformer.from_dataset(
    train_ds,
    learning_rate=1e-3,
    hidden_size=6,
    dropout=0.4,
    loss=nnse(),
    reduce_on_plateau_patience=6,
    optimizer='adam')

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
