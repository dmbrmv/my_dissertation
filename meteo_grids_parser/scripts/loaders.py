from pathlib import Path
import glob


def multi_var_nc(path_to_nc: Path):
    # get variables for computation
    var_names = [i.split('/')[-1]
                 for i in
                 glob.glob(f'{path_to_nc}/*')]

    # define paths
    data_paths = {var: glob.glob(f'{path_to_nc}/{var}/*.nc')
                  for var in var_names}

    return data_paths


def aggregation_definer(dataset: str,
                        variable: str):

    if dataset == 'gleam':
        return 'sum'
    elif ('precipitation' in variable) | ('evaporation' in variable):
        return 'sum'
    else:
        return 'mean'
