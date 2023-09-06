from pathlib import Path
import glob
import gc
import torch
import random
import geopandas as gpd
from neuralhydrology.nh_run import start_run
from scripts.file_manipulator import train_rewriter
from neuralhydrology.utils.config import Config


era_input = ['prcp_e5l',  't_max_e5l', 't_min_e5l']
# q_mm_day or lvl_mbs
hydro_target = 'q_mm_day'
q_h_relation = False

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
    if q_h_relation:
        nc_variable = 'nc_all_q_h'
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

ts_dir = Path('../geo_data/time_series')

# time series directory
ts_dir = Path('../geo_data/time_series')
# write files for train procedure
print(f'train data for {hydro_target} with {nc_variable} initial data')
train_rewriter(era_pathes=glob.glob(
    f'../geo_data/great_db/{nc_variable}/*.nc'),
               ts_dir=ts_dir,
               hydro_target=hydro_target,
               area_index=ws_file.index,
               predictors=era_input)

# define variables require to perform hindcast
gauges = [file.split('/')[-1][:-3] for
          file in glob.glob(f'{ts_dir}/*.nc')]
random.shuffle(gauges)
gauge_size = len(gauges)


cfg = Config(Path('./model_config.yml'))
# base model type
# [cudalstm, customlstm, ealstm, embcudalstm, mtslstm, gru, transformer]
# (has to match the if statement in modelzoo/__init__.py)
model_name = 'cudalstm'
cfg.update_config(yml_path_or_dict={
    # define storage and experiment
    'experiment_name': f'{model_name}_{hydro_target}',
    'model': f'{model_name}',
    'run_dir': './',
    'data_dir': '../geo_data/',
    # define inner parameters
    'static_attributes': static_parameters,
    'dynamic_inputs': era_input,
    # 'hindcast_inputs': era_input,
    # 'forecast_inputs': era_input,
    'target_variables': [hydro_target],
    # 'dynamics_embedding': {'type': 'fc', 'hiddens': [128, 64, 256],
    #                       'activation': 'tanh', 'dropout': 0.2},
    # 'statics_embedding': {'type': 'fc', 'hiddens': [128, 64, 256],
    #                      'activation': 'tanh', 'dropout': 0.2},
    # define files with gauge data
    'train_basin_file': './every_basin.txt',
    'validate_n_random_basins': gauge_size,
    'validation_basin_file': './every_basin.txt',
    'test_basin_file': './every_basin.txt',
    # define time periods
    # 'seq_length': 14,
    # 'forecast_seq_length': 10,
    'train_start_date': '01/01/2009',
    'train_end_date': '31/12/2016',
    'validation_start_date': '01/01/2017',
    'validation_end_date': '31/12/2018',
    'test_start_date': '01/01/2019',
    'test_end_date': '31/12/2020'})
cfg.dump_config(folder=Path('./launch_configs'),
                filename=f'{model_name}_{hydro_target}.yml')

gc.collect()
if torch.cuda.is_available():
    start_run(config_file=Path(
        f'./launch_configs/{model_name}_{hydro_target}.yml'))
