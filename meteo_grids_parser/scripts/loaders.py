from pathlib import Path
import glob


def multi_var_nc(path_to_nc: Path,
                 file_extension: str) -> dict:
    # get variables for computation
    var_names = [i.split('/')[-1]
                 for i in
                 glob.glob(f'{path_to_nc}/*')]

    # define paths
    data_paths = {var: glob.glob(f'{path_to_nc}/{var}/*.{file_extension}')
                  for var in var_names}

    return data_paths


def aggregation_definer(dataset: str,
                        variable: str):

    if dataset == 'gleam':
        return 'sum'
    elif (('precipitation' in variable) |
          ('evaporation' in variable) | ('tot_prec' in variable)):
        return 'sum'
    else:
        return 'mean'


def grid_descriptor(dataset_name: str,
                    half_resolution: float,
                    files: Path,
                    extension: str = 'nc'):
    """_summary_

    Args:
        dataset_name (str): _description_
        half_resolution (float): _description_
        files (Path): _description_
        extension (str, optional): _description_. Defaults to 'nc'.

    Returns:
        _type_: _description_
    """

    return {dataset_name: {'res': half_resolution,
                           'f_path': multi_var_nc(path_to_nc=files,
                                                  file_extension=extension)}}
