from scripts.tft_data import open_for_tft, train_val_split_std
# from scripts.model_eval import nnse

import glob
import geopandas as gpd

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
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
# q_mm_day or lvl_mbs
hydro_target = 'lvl_mbs'

if hydro_target == 'lvl_mbs':
    static_parameters = ['for_pc_sse', 'crp_pc_sse',
                         'inu_pc_ult', 'ire_pc_sse',
                         'lka_pc_use', 'prm_pc_sse',
                         'pst_pc_sse', 'cly_pc_sav',
                         'slt_pc_sav', 'snd_pc_sav',
                         'kar_pc_sse', 'urb_pc_sse',
                         'gwt_cm_sav', 'lkv_mc_usu',
                         'rev_mc_usu', 'sgr_dk_sav',
                         'slp_dg_sav', 'ws_area',
                         'ele_mt_sav', 'height_bs']
    nc_variable = 'nc_all_h'
else:
    static_parameters = ['for_pc_sse', 'crp_pc_sse',
                         'inu_pc_ult', 'ire_pc_sse',
                         'lka_pc_use', 'prm_pc_sse',
                         'pst_pc_sse', 'cly_pc_sav',
                         'slt_pc_sav', 'snd_pc_sav',
                         'kar_pc_sse', 'urb_pc_sse',
                         'gwt_cm_sav', 'lkv_mc_usu',
                         'rev_mc_usu', 'sgr_dk_sav',
                         'slp_dg_sav', 'ws_area',
                         'ele_mt_sav']
    nc_variable = 'nc_all_q'

ws_file = gpd.read_file('../geo_data/great_db/geometry/russia_ws.gpkg')
ws_file = ws_file.set_index('gauge_id')

file = open_for_tft(
    nc_files=glob.glob(f'../geo_data/great_db/{nc_variable}/*.nc'),
    static_path='../geo_data/attributes/static_with_height.csv',
    area_index=ws_file.index,
    meteo_input=meteo_input,
    hydro_target=hydro_target,
    with_static=True,
    shuffle_static=False)

train_val_data = train_val_split_std(file,
                                     hydro_target=hydro_target,
                                     with_static=True)


# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-3, patience=3, verbose=True,
                                    mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
# logging results to a tensorboard
logger = TensorBoardLogger("h_prediction_std_meteo_encoder")

if device == 'cuda':
    accel = 'gpu'
else:
    accel = 'cpu'

trainer = pl.Trainer(
    max_epochs=20,
    accelerator='auto',
    enable_model_summary=True,
    check_val_every_n_epoch=1,
    gradient_clip_val=0.4,
    log_every_n_steps=1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,)

tft = TemporalFusionTransformer.from_dataset(
    train_val_data['train_ds'],
    learning_rate=1e-3,
    hidden_size=256,
    hidden_continuous_size=128,
    dropout=0.4,
    lstm_layers=2,
    attention_head_size=8,
    loss=QuantileLoss(),
    # time_varying_reals_encoder=meteo_input,
    # allowed_encoder_known_variable_names=meteo_input,
    optimizer='adam')

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# fit network
trainer.fit(tft,
            train_dataloaders=train_val_data['train_dl'],
            val_dataloaders=train_val_data['val_dl'])
