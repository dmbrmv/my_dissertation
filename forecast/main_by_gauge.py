from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import RMSE
import pytorch_lightning as pl
from scripts.tft_data import open_for_tft, train_val_split
# from scripts.model_eval import nnse, pred_res_builder

import glob
import geopandas as gpd
# import pandas as pd

import torch
torch.set_float32_matmul_precision('medium')


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
hydro_target = 'lvl_mbs'
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


ws_file = gpd.read_file('../geo_data/great_db/geometry/russia_ws.gpkg')
ws_file = ws_file.set_index('gauge_id')
tft_gauges = [f.split('/')[-1][:-4]
              for f in glob.glob('./single_gauge_level_30epoch/*')]
print(tft_gauges)

for nc_file in glob.glob('../geo_data/great_db/nc_all_h/*.nc'):
    try:
        gauge_id = nc_file.split('/')[-1][:-3]
        if gauge_id in tft_gauges:
            pass
        else:
            file = open_for_tft(
                nc_files=[nc_file],
                static_path='../geo_data/attributes/geo_vector.csv',
                area_index=ws_file.index,
                meteo_input=meteo_input,
                hydro_target=hydro_target, allow_nan=True)

            (train_ds, train_loader,
                val_ds, val_loader, val_df,
                scaler) = train_val_split(file)

            # configure network and trainer
            early_stop_callback = EarlyStopping(monitor="val_loss",
                                                min_delta=1e-3, patience=3,
                                                verbose=True, mode="min")
            # log the learning rate
            lr_logger = LearningRateMonitor()
            # logging results to a tensorboard
            logger = TensorBoardLogger(
                f"./single_gauge_level_30epoch/{gauge_id}_tft")

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
                hidden_size=256,
                dropout=0.4,
                loss=RMSE(),
                optimizer='adam')

        # print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

            # fit network
            trainer.fit(tft,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader)
    except Exception as e:
        with open('error_file.txt', 'a') as f:
            f.write(''.join(f'{e} -- for gauge {gauge_id}\n'))

# by_gauge_res = list()
# for nc_file in glob.glob('../geo_data/great_db/nc_all_q/*.nc'):
#     gauge_id = nc_file.split('/')[-1][:-3]
#     try:
#         file = open_for_tft(
#             nc_files=[nc_file],
#             static_path='../geo_data/attributes/geo_vector.csv',
#             area_index=ws_file.index,
#             meteo_predictors=meteo_input,
#             hydro_target=hydro_target, allow_nan=True)

#         (_, _,
#          val_ds, _, val_df,
#          scaler) = train_val_split(file)

#         chkpt = glob.glob(
#             f'./single_gauge_8epoch/{gauge_id}_tft/*/*/checkpoints/*.ckpt')[0]
#         resdf, _ = pred_res_builder(
#             gauge_id=gauge_id,
#             hydro_target=hydro_target,
#             meteo_input=meteo_input,
#             static_parameters=static_parameters,
#             model_checkpoint=chkpt,
#             res_storage='./result/tft_single_8epoch',
#             val_df=val_df,
#             scaler=scaler,
#             val_ts_ds=val_ds, with_plot=False)
#         by_gauge_res.append(resdf)
#     except ValueError:
#         continue

# by_gauge_res = pd.concat(by_gauge_res)
# by_gauge_res.to_csv('./result/tft_by_gauge_4static.csv', index=False)
