import pandas as pd


def split_by_hydro_year(discharge_obs: pd.Series):
    return {str(year): discharge_obs[f'10/01/{year}':f'10/01/{year+1}']
            for year in discharge_obs.index.year.unique()}


def split_by_year(discharge_obs: pd.Series):
    return {str(year): discharge_obs[f'01/01/{year}':f'12/31/{year}']
            for year in discharge_obs.index.year.unique()}
