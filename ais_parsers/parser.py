import glob
from pathlib import Path
from tqdm import tqdm
from scripts.ais_parsers import level_to_csv, discharge_to_csv
import pandas as pd

h_path = glob.glob('../geo_data/ais_gmvo_16_03_23/levels/*.xlsx')

id_lbl_asso = list()
for f_path in tqdm(h_path):

    id_lbl_asso.append(level_to_csv(data_path=f_path,
                                    from_top=39,
                                    save_path=Path('./data/res_levels/')))

temp_df = list()
for record in id_lbl_asso:
    temp_df.append(pd.DataFrame(record).T)
temp_df = pd.concat(temp_df)
temp_df = temp_df.rename(columns={0: 'name',
                                  1: 'height'})
temp_df.to_csv('./data/height_id.csv')


q_path = glob.glob('../geo_data/ais_gmvo_16_03_23/discharge/*.xlsx')

q_lbl_asso = list()
for i, f_path in enumerate(tqdm(q_path)):
    print(f'{f_path}, -- {i+1}')
    q_lbl_asso.append(
        discharge_to_csv(data_path=f_path,
                         save_path=Path('./data/res_discharge/')))
print(q_lbl_asso)
