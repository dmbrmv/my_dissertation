import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
import torch

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


def file_checker(file_path: str,
                 meteo_predictors: list,
                 hydro_target: str):
    condition = xr.open_dataset(
        file_path).to_dataframe()[
            [hydro_target,
             *meteo_predictors]].isna().sum().sum() != 0

    return condition


def open_for_tft(nc_files: list,
                 static_path: str,
                 meteo_predictors: list,
                 area_index,
                 hydro_target: str,
                 index_col: str = 'gauge_id') -> pd.DataFrame:

    static_attributes = pd.read_csv(static_path,
                                    index_col=index_col)
    static_attributes.index = static_attributes.index.astype(str)

    res_file = list()
    for file_path in tqdm(nc_files):
        gauge_id = file_path.split('/')[-1][:-3]
        if gauge_id in really_bad_gauges:
            pass
        else:
            try:
                gauge_static = static_attributes.loc[[gauge_id], :]
            except KeyError:
                print(f'No data for {gauge_id} !')
                continue
            cond = file_checker(file_path=file_path,
                                meteo_predictors=meteo_predictors,
                                hydro_target=hydro_target)
            if cond:
                continue
            elif gauge_id not in area_index:
                pass
            else:
                file = xr.open_dataset(file_path)
                file = file.to_dataframe()
                # file['date'] = file.index
                file = file.reset_index()
                file['time_idx'] = file.index

                for col in gauge_static.columns:
                    file[col] = gauge_static.loc[gauge_id, col]

                res_file.append(file)

    file = pd.concat(res_file, axis=0)
    file = file.reset_index(drop=True)

    return file


def train_val_split(big_df: pd.DataFrame,
                    meteo_input: list = [
                        'prcp_e5l',  't_max_e5l', 't_min_e5l'],
                    hydro_target: str = 'q_mm_day',
                    static_parameters: list = ['for_pc_sse', 'crp_pc_sse',
                                               'inu_pc_ult', 'ire_pc_sse',
                                               'lka_pc_use', 'prm_pc_sse',
                                               'pst_pc_sse', 'cly_pc_sav',
                                               'slt_pc_sav', 'snd_pc_sav',
                                               'kar_pc_sse', 'urb_pc_sse',
                                               'gwt_cm_sav', 'lkv_mc_usu',
                                               'rev_mc_usu', 'sgr_dk_sav',
                                               'slp_dg_sav', 'ws_area',
                                               'ele_mt_sav'],
                    encoder_length: int = 365,
                    prediction_length: int = 7,
                    train_end: str = '2017-12-31',
                    train_start: str = '2008-01-01',
                    batch_size: int = 256):

    big_df = big_df[['date', 'time_idx', 'gauge_id',
                     hydro_target, *meteo_input, *static_parameters]]
    big_df = big_df.dropna().reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(1, 10))
    big_df[[hydro_target, *meteo_input,
            *static_parameters]] = scaler.fit_transform(
        big_df[[hydro_target, *meteo_input, *static_parameters]])

    train_df = big_df[lambda x:
                      (train_start <= x.date) &
                      (x.date <= train_end)].reset_index(
                          drop=True)  # type: ignore
    # set time_idx in according to slice
    # with range from 0 to max for each gauge
    train_df['time_idx'] -= train_df['time_idx'].min()
    val_df = big_df[lambda x:
                    x.date > train_end].reset_index(drop=True)  # type: ignore
    # set time_idx in according to slice
    # with range from 0 to max for each gauge
    val_df['time_idx'] -= val_df['time_idx'].min()

    train_ds = TimeSeriesDataSet(
        data=train_df,
        time_idx="time_idx",
        target=hydro_target,
        group_ids=["gauge_id"],
        time_varying_unknown_reals=[hydro_target],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_reals=static_parameters,
        time_varying_known_reals=meteo_input,
        scalers={'name': 'None'})

    val_ds = TimeSeriesDataSet.from_dataset(train_ds,
                                            val_df)

    train_dataloader = train_ds.to_dataloader(train=True,
                                              batch_size=batch_size,
                                              num_workers=8)
    val_dataloader = val_ds.to_dataloader(train=False,
                                          batch_size=batch_size,
                                          num_workers=8)

    return train_ds, train_dataloader, val_ds, val_dataloader


class nnse(MultiHorizonMetric):

    def loss(self, pred, target):

        pred = self.to_prediction(pred)
        denom = torch.sum((target-pred)**2)
        divsr = torch.sum((target - torch.mean(target)**2))
        nse = 1 - torch.div(denom, divsr)
        nnse = 1 / (2 - nse)

        return nnse
