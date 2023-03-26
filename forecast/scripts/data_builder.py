import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr
import sys
sys.path.append('/workspaces/my_dissertation')
from input_creator.scripts.readers import xr_opener


class HydroForecastDataset(Dataset):

    def __init__(self, data_dict):
        self.data = data_dict

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.Tensor(x['past_seq']), torch.Tensor(x['future_seq'])

    def __len__(self):
        return len(self.data)


class HydroForecastData():
    """
    Return torch read Dataset in shape of:
    [T]ime, [V]ariables, [G]auges, [I]nterval
    """

    def __init__(self,
                 nc_files: list, gauges: list,
                 predictors: list, target: list,
                 hydroatlas: list, hydroatlas_path: str, h_bs_file: str,
                 future_interval: int, past_interval: int,
                 train_start: str, train_end: str,
                 val_start: str, val_end: str,
                 test_start: str, test_end: str):
        """_summary_

        Args:
            nc_files (list): glob.glob of all available nc files with 
            hydrometeo attributes
            gauges (list): list of str representations of gauge id's

            predictors (list): possible meteo predictors:
            ['t_max_e5l', 't_max_e5', 't_min_e5l', 't_min_e5',
            'prcp_e5l', 'prcp_e5', 'prcp_gpcp', 'prcp_imerg',
            'prcp_mswep', 'Eb', 'Es', 'Et', 'SMsurf', 'SMroot',
            'Ew', 'Ei', 'S', 'E', 'Ep']

            target (list): possible hydrological targets:
            ['lvl_sm', 'q_cms_s', 'lvl_mbs', 'q_mm_day']

            hydroatlas (list): list of hydro atlas attributes

            hydroatlas_path (str): path to precalculated HydroATLAS attributes
            stored in single file

            h_bs_file (str): path to file with association
            of height in meters of baltic system to gauge id

            future_interval (int): window of future targets prediction

            past_interval (int): window of past predictors before prediction

            train_start (str): date of train start e.g. mm/dd/yyyy
            train_end (str): date of train end

            val_start (str): date of val start e.g. mm/dd/yyyy
            val_end (str): date of val end e.g. mm/dd/yyyy

            test_start (str): date of test start e.g. mm/dd/yyyy
            test_end (str): date of test end e.g. mm/dd/yyyy
        """

        big_ds = xr.concat([xr_opener(file) for file in nc_files],
                           dim='gauge_id')
        self.gauges = gauges
        # select dataset with only selected gauges
        self.ds = big_ds.sel(gauge_id=self.gauges)
        big_ds.close()
        self.predictors = predictors
        self.target = target
        self.future_interval = future_interval
        self.past_interval = past_interval
        self.train_start, self.train_end = train_start, train_end
        self.val_start, self.val_end = val_start, val_end
        self.test_start, self.test_end = test_start, test_end

        # hydro atlas read
        ha_df = pd.read_csv(hydroatlas_path,
                            index_col='gauge_id')
        ha_df.index = ha_df.index.astype(str)
        # add height of gauge in m bs
        height_bs = pd.read_csv(h_bs_file,
                                index_col='gauge_id')
        height_bs.index = height_bs.index.astype(str)
        # select static attributes
        ha_df = ha_df.loc[self.gauges, hydroatlas]
        # add correspond height
        ha_df['m_bs'] = height_bs.loc[gauges, ['height']]
        self.static_attributes = torch.Tensor(ha_df.to_numpy())

    def split_ds(self,
                 dataset: xr.Dataset) -> dict:
        self.data = dict()
        for i in range(self.past_interval,
                       len(dataset['date'])-self.future_interval):

            t = dataset['date'].isel(date=i).values
            t = pd.to_datetime(t).strftime('%Y-%m-%d')

            self.data[i-self.past_interval] = {
                'past_seq': dataset[
                    self.predictors].to_array().values[:, :,
                                                       i-self.past_interval:i],
                'future_seq': dataset[
                    self.target].to_array().values[:, :,
                                                   i:i+self.future_interval]
            }

        return self.data

    def train_val_test(self):

        train_ds = self.split_ds(
            dataset=self.ds.sel(date=slice(self.train_start, self.train_end)))
        val_ds = self.split_ds(
            dataset=self.ds.sel(date=slice(self.val_start, self.val_end)))
        test_ds = self.split_ds(
            dataset=self.ds.sel(date=slice(self.test_start, self.test_end)))

        return (HydroForecastDataset(train_ds),
                HydroForecastDataset(val_ds),
                HydroForecastDataset(test_ds))
