import glob
from pathlib import Path
from tqdm import tqdm
from scripts.ais_parsers import discharge_to_csv, level_to_csv, file_extender
import pandas as pd

h_path = glob.glob('../geo_data/great_db/ais_data/levels_xls/*.xls')

id_lbl_asso = list()
for f_path in tqdm(h_path):
    h_name = f_path.split('/')[-1][:-4]
    id_lbl_asso.append(level_to_csv(data_path=f_path,
                                    save_path=Path(
                                        f'./data_2/lvl_csv/{h_name}')))

temp_df = list()
for record in id_lbl_asso:
    temp_df.append(pd.DataFrame(record).T)
temp_df = pd.concat(temp_df)
temp_df = temp_df.rename(columns={0: 'name',
                                  1: 'height'})
temp_df.to_csv('./data_2/height_id.csv')

file_extender(files_by_district=glob.glob('./data_2/lvl_district_csv/*/*.csv'),
              save_storage=Path('./data_2/res/levels'))

q_path = glob.glob('../geo_data/great_db/ais_data/discharge_xls/*.xls')

q_lbl_asso = list()
for i, f_path in enumerate(tqdm(q_path)):
    print(f'{f_path}, -- {i+1}')
    q_name = f_path.split('/')[-1][:-4]
    q_lbl_asso.append(
        discharge_to_csv(data_path=f_path,
                         save_path=Path(
                             f'./data_2/q_csv/{q_name}')))

file_extender(files_by_district=glob.glob('./data_2/q_district_csv/*/*.csv'),
              save_storage=Path('./data_2/res/discharges'))
