import glob
from pathlib import Path
from tqdm import tqdm
from scripts.ais_parsers import level_to_csv
import pandas as pd

file_path = glob.glob('./data/initial_levels/converted_1/*.xlsx')

id_lbl_asso = list()
for f_path in tqdm(file_path):

    id_lbl_asso.append(level_to_csv(data_path=f_path,
                                    from_top=38,
                                    save_path=Path('./data/levels/results/')))

temp_df = list()
for record in id_lbl_asso:
    temp_df.append(pd.DataFrame(record).T)
temp_df = pd.concat(temp_df)
temp_df = temp_df.rename(columns={0: 'name',
                                  1: 'height'})
temp_df.to_csv('./data/levels/height_id.csv')


file_path = glob.glob('./data/initial_levels/*.xlsx')

id_lbl_asso = list()
for i, f_path in enumerate(tqdm(file_path)):
    print(f'{f_path}, -- {i+1}')
    id_lbl_asso.append(
        level_to_csv(data_path=f_path,
                     from_top=39,
                     save_path=Path('./data/levels/new_results/')))
