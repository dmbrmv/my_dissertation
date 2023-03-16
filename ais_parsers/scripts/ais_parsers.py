import pandas as pd
import numpy as np
import re
from pathlib import Path


def df_from_excel(observations: np.ndarray,
                  dates: pd.DatetimeIndex,
                  col_name: str):
    temp_df = pd.DataFrame()

    temp_df['date'] = dates
    temp_df[col_name] = observations
    temp_df[col_name] = temp_df[col_name].replace(to_replace=-9999,
                                                  value=np.NaN)
    return temp_df


def replace_val(my_val):

    if my_val == '' or my_val == '-' or my_val == '--' or my_val == ' ':
        return -9999
    elif isinstance(my_val, str):
        if my_val[-1] == '-':
            my_val = my_val[:-1]
        try:
            return pd.to_numeric(my_val)
        except ValueError:
            pass
    else:
        return np.NaN


def discharge_to_csv(data_path: str,
                     save_path: Path) -> dict:
    """Script allows to parse data from AIS GMVO (https://gmvo.skniivh.ru/)
    Excel output format both for level and discharge into .csv for separate
    observation points included in final file

    Args:
        data_path (Path): Path to .xlsx file
        save_path (Path): Folder where results will be stored
    """

    save_path.mkdir(exist_ok=True, parents=True)

    month_days = 31
    # if 0 -- ID of gauge, 2 -- river name
    river_step = 0
    river_label = 2
    year_step = 1
    m_bs = 3

    skip_top = 18
    month_step = 5
    table_step = 53

    file = pd.read_excel(data_path, skiprows=skip_top, skipfooter=0)

    monthes_range = list(range(month_step, file.shape[0], table_step))
    river_fields = list(range(river_step, file.shape[0], table_step))
    river_labels = list(range(river_label, file.shape[0], table_step))
    m_bs_fields = list(range(m_bs, file.shape[0], table_step))
    year_fields = list(range(year_step, file.shape[0], table_step))

    river_ids = np.array([file.iloc[river_fields[i], 1]
                          for i in range(len(monthes_range))])

    river_names = np.array([file.iloc[river_labels[i], 1]
                            for i in range(len(monthes_range))])

    m_bs_values = np.array([file.iloc[m_bs_fields[i], 1]
                            for i in range(len(monthes_range))])
    # ID always unique
    number_of_rivers = len(np.unique(river_ids))
    # define number of downloaded rivers
    results = {river: [] for river in river_ids[0:number_of_rivers]}

    for i, (river_name, data) in enumerate(results.items()):

        test_selection = file.iloc[
            monthes_range[i]:monthes_range[i] + month_days, 1:]

        test_selection = test_selection.applymap(
            lambda x: str(x).replace('прсх', '0')).applymap(
            lambda x: str(x).replace('прмз', '0')).applymap(
            lambda x: str(x).replace(',', '.')).applymap(
            lambda x: str(x).replace('?', '')).applymap(
            lambda x: re.sub('[^-?0-9.]', '', x))

        dates = pd.date_range(
            start=f'{file.iloc[year_fields[i], 1]}-01-01',
            end=f'{file.iloc[year_fields[i], 1]}-12-31')

        if len(dates) == 365:
            test_selection.loc[:,
                               'Unnamed: 2'].values[-3:] = np.array(
                [np.NaN, np.NaN, np.NaN])
        elif len(dates) == 366:
            if test_selection.loc[:,
                                  'Unnamed: 2'].values[-3] == '':
                try:
                    row_mean = test_selection.loc[
                        test_selection.index[:-3],
                        'Unnamed: 2'].astype(int).mean()
                    test_selection.loc[:,
                                       'Unnamed: 2'].values[-3] = row_mean
                    test_selection.loc[
                        :, 'Unnamed: 2'].values[-2:] = np.array([np.NaN,
                                                                 np.NaN])
                except ValueError:
                    test_selection.loc[
                        :, 'Unnamed: 2'].values[-2:] = np.array([np.NaN,
                                                                 np.NaN])
            else:
                test_selection.loc[:,
                                   'Unnamed: 2'].values[-2:] = np.array(
                    [np.NaN, np.NaN])

        test_selection.loc[:,
                           'Unnamed: 4'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 6'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 9'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 11'].values[-1:] = np.array(
            [np.NaN])

        if '.' in test_selection.values:
            rows = np.flatnonzero(
                (test_selection == '.').values)//test_selection.shape[1]
            cols = np.flatnonzero(
                (test_selection == '.').values) % test_selection.shape[1]

            prev_vals = list()
            for r, c in zip(rows, cols):
                try:
                    prev_vals.append(pd.to_numeric(
                        test_selection.iloc[r-1, c]))
                except IndexError:
                    prev_vals.append(pd.to_numeric(
                        test_selection.iloc[r+1, c]))

            next_vals = list()
            for r, c in zip(rows, cols):
                try:
                    next_vals.append(pd.to_numeric(
                        test_selection.iloc[r+1, c]))
                except IndexError:
                    next_vals.append(pd.to_numeric(
                        test_selection.iloc[r-1, c]))

            fill_val = [np.mean([pr, nx])
                        for pr, nx in zip(prev_vals, next_vals)]

            for i, (r, c) in enumerate(zip(rows, cols)):
                test_selection.iat[r, c] = str(fill_val[i])

            test_selection = np.array(list(map(
                replace_val,
                test_selection.to_numpy().T.flatten())),
                dtype=float)
        else:
            test_selection = np.array(list(map(
                replace_val,
                test_selection.to_numpy().T.flatten())),
                dtype=float)

        test_selection = test_selection[~np.isnan(test_selection)]

        # ID always unique -- no need to check river name

        try:
            results[river_name].append(
                df_from_excel(observations=test_selection,
                              dates=dates,
                              col_name='discharge'))
        except ValueError:
            print(
                f'{river_name} at {file.iloc[year_fields[i], 1]}\n')
            print(data_path)
            print('\n')

    label_id = {}
    # association between id and river name
    for i, r_id in enumerate(river_ids):
        if r_id in label_id.keys():
            pass
        else:
            label_id[r_id] = [river_names[i], m_bs_values[i]]

    for river_name, data in results.items():
        data = pd.concat(data)

        data.to_csv(f'{save_path}/{river_name}.csv')

    return label_id


def level_to_csv(data_path: str,
                 save_path: Path,
                 from_top: int = 38) -> dict:
    """Script allows to parse data from AIS GMVO (https://gmvo.skniivh.ru/)
    Excel output format both for level and discharge into .csv for separate
    observation points included in final file

    Args:
        data_path (Path): Path to .xlsx file
        save_path (Path): Folder where results will be stored
    """

    save_path.mkdir(exist_ok=True, parents=True)

    month_days = 31

    # if 0 -- ID of gauge, 2 -- river name
    river_step = 0
    river_label = 2
    year_step = 1
    m_bs = 3

    skip_top = from_top
    month_step = 7
    table_step = 55

    file = pd.read_excel(data_path, skiprows=skip_top, skipfooter=0)

    monthes_range = list(range(month_step, file.shape[0], table_step))
    river_fields = list(range(river_step, file.shape[0], table_step))
    river_labels = list(range(river_label, file.shape[0], table_step))
    m_bs_fields = list(range(m_bs, file.shape[0], table_step))
    year_fields = list(range(year_step, file.shape[0], table_step))

    river_ids = np.array([file.iloc[river_fields[i], 1]
                          for i in range(len(monthes_range))])

    river_names = np.array([file.iloc[river_labels[i], 1]
                            for i in range(len(monthes_range))])

    m_bs_values = np.array([file.iloc[m_bs_fields[i], 1]
                            for i in range(len(monthes_range))])
    # ID always unique
    number_of_rivers = len(np.unique(river_ids))
    # define number of downloaded rivers
    results = {river: [] for river in river_ids[0:number_of_rivers]}

    for i, (river_name, data) in enumerate(results.items()):

        test_selection = file.iloc[
            monthes_range[i]:monthes_range[i] + month_days, 1:]

        test_selection = test_selection.applymap(
            lambda x: str(x).replace('прсх', '0')).applymap(
            lambda x: str(x).replace('прмз', '0')).applymap(
            lambda x: str(x).replace(',', '.')).applymap(
            lambda x: str(x).replace('?', '')).applymap(
            lambda x: re.sub('[^-?0-9.]', '', x))

        dates = pd.date_range(
            start=f'{file.iloc[year_fields[i], 1]}-01-01',
            end=f'{file.iloc[year_fields[i], 1]}-12-31')

        if len(dates) == 365:
            test_selection.loc[:,
                               'Unnamed: 2'].values[-3:] = np.array(
                [np.NaN, np.NaN, np.NaN])
        elif len(dates) == 366:
            if test_selection.loc[:,
                                  'Unnamed: 2'].values[-3] == '':
                try:
                    row_mean = test_selection.loc[
                        test_selection.index[:-3],
                        'Unnamed: 2'].astype(int).mean()
                    test_selection.loc[:,
                                       'Unnamed: 2'].values[-3] = row_mean
                    test_selection.loc[
                        :, 'Unnamed: 2'].values[-2:] = np.array([np.NaN,
                                                                 np.NaN])
                except ValueError:
                    test_selection.loc[
                        :, 'Unnamed: 2'].values[-2:] = np.array([np.NaN,
                                                                 np.NaN])
            else:
                test_selection.loc[:,
                                   'Unnamed: 2'].values[-2:] = np.array(
                    [np.NaN, np.NaN])

        test_selection.loc[:,
                           'Unnamed: 4'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 6'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 9'].values[-1:] = np.array(
            [np.NaN])
        test_selection.loc[:,
                           'Unnamed: 11'].values[-1:] = np.array(
            [np.NaN])

        if '.' in test_selection.values:
            rows = np.flatnonzero(
                (test_selection == '.').values)//test_selection.shape[1]
            cols = np.flatnonzero(
                (test_selection == '.').values) % test_selection.shape[1]

            prev_vals = list()
            for r, c in zip(rows, cols):
                try:
                    prev_vals.append(pd.to_numeric(
                        test_selection.iloc[r-1, c]))
                except IndexError:
                    prev_vals.append(pd.to_numeric(
                        test_selection.iloc[r+1, c]))

            next_vals = list()
            for r, c in zip(rows, cols):
                try:
                    next_vals.append(pd.to_numeric(
                        test_selection.iloc[r+1, c]))
                except IndexError:
                    next_vals.append(pd.to_numeric(
                        test_selection.iloc[r-1, c]))

            fill_val = [np.mean([pr, nx])
                        for pr, nx in zip(prev_vals, next_vals)]

            for i, (r, c) in enumerate(zip(rows, cols)):
                test_selection.iat[r, c] = str(fill_val[i])

            test_selection = np.array(list(map(
                replace_val,
                test_selection.to_numpy().T.flatten())),
                dtype=float)
        else:
            test_selection = np.array(list(map(
                replace_val,
                test_selection.to_numpy().T.flatten())),
                dtype=float)

        test_selection = test_selection[~np.isnan(test_selection)]
        try:
            results[river_name].append(
                df_from_excel(observations=test_selection,
                              dates=dates,
                              col_name='level'))
        except ValueError:
            print(f'\n{river_name} at {file.iloc[year_fields[i], 1]}')
            print(data_path)
            print('\n')

    label_id = {}
    # association between id and river name
    for i, r_id in enumerate(river_ids):
        if r_id in label_id.keys():
            pass
        else:
            label_id[r_id] = [river_names[i], m_bs_values[i]]

    for river_name, data in results.items():
        data = pd.concat(data)

        data.to_csv(f'{save_path}/{river_name}.csv')

    return label_id
