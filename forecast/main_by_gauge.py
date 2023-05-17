from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from scripts.tft_data import open_for_tft, train_val_split
from scripts.model_eval import nnse, pred_res_builder

import glob
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

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
hydro_target = 'q_mm_day'
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
really_bad_gauges = ['48011', '72004', '12614', '12305', '4036', '4226',
                     '5190', '10357', '80174', '10549', '9344', '10688',
                     '84448', '74425', '78507', '84803', '8256', '8359',
                     '5607', '84108', '72246', '8139', '10539', '75527',
                     '4197', '78343', '75428', '84200', '19383', '78225',
                     '84213', '19135', '12354', '9331', '72518', '4068',
                     '84398', '2164', '5613', '77362', '8120', '4012',
                     '5352', '84043', '84453', '8281', '48008', '83260',
                     '78501', '84119', '76145', '10058', '84245', '4235',
                     '84295', '75780', '72450', '5199', '9309', '48056',
                     '75110', '4240', '77164', '84019', '4005', '5611',
                     '84157', '842000', '49128', '9546', '5604', '76325',
                     '9511', '10060', '49095', '84315', '72198', '77327',
                     '78620', '78163', '84185', '76644', '9345', '7171',
                     '4071', '84400', '19111', '19008', '6563', '75271',
                     '76470', '70129', '72744', '9355', '11327', '70102',
                     '11374', '75222', '19167', '75387', '78261',
                     '10584', '78499', '84354', '10548', '11675']

ws_file = gpd.read_file('../geo_data/great_db/geometry/russia_ws.gpkg')
ws_file = ws_file.set_index('gauge_id')
ws_file = ws_file[ws_file['new_area'] <= 50000]

by_gauge_res = list()
for nc_file in glob.glob('../geo_data/great_db/nc_all_q/*.nc'):
    gauge_id = nc_file.split('/')[-1][:-3]
    if gauge_id in really_bad_gauges:
        pass
    else:
        try:
            file = open_for_tft(
                nc_files=[nc_file],
                static_path='../geo_data/attributes/geo_vector.csv',
                area_index=ws_file.index,
                meteo_predictors=meteo_input,
                hydro_target=hydro_target)

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
            logger = TensorBoardLogger(f"./single_gauge/{gauge_id}_tft")

            if device == 'cuda':
                accel = 'gpu'
            else:
                accel = 'cpu'

            trainer = pl.Trainer(
                max_epochs=15,
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
                hidden_size=64,
                dropout=0.4,
                loss=nnse(),
                optimizer='adam')

            # print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

            # fit network
            trainer.fit(tft,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader)
            chkpt = glob.glob(
                f'./single_gauge/{gauge_id}_tft/*/*/checkpoints/*.ckpt')[0]
            resdf, _ = pred_res_builder(gauge_id=gauge_id,
                                        hydro_target=hydro_target,
                                        meteo_input=meteo_input,
                                        static_parameters=static_parameters,
                                        model_checkpoint=chkpt,
                                        val_df=val_df,
                                        scaler=scaler,
                                        val_ts_ds=val_ds, with_plot=False)
            by_gauge_res.append(resdf)
        except Exception as e:
            with open('error_file.txt', 'a') as f:
                f.write(''.join(f'{e} -- for gauge {gauge_id}'))
by_gauge_res = pd.concat(by_gauge_res)
by_gauge_res.to_csv('./result/tft_by_gauge.csv', index=False)

