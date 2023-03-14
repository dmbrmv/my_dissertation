from tqdm import tqdm
import geopandas as gpd
from scripts.feature_selection import featureXtractor, save_results

# set path to shape file
home_folder = '/home/anton/dima_experiments/my_dissertation/geo_data'
# path to geodataframe
shape_file_path = f'{home_folder}/geometry/russia_ws.gpkg'
# set path to downloaded HydroATLAS
gdb_file_path = f'{home_folder}/hydro_atlas/BasinATLAS_v10.gdb'
# set path where results will be stored
path_to_save = f'{home_folder}/static_attributes'

# Read shape file with geometry column
my_shape = gpd.read_file(shape_file_path)

output = list()
for ws in tqdm(my_shape.loc[:, 'geometry']):
    output.append(featureXtractor(user_ws=ws,
                                  gdb_file_path=gdb_file_path))

save_results(extracted_data=output,
             gauge_ids=my_shape.loc[:, 'gauge_id'],
             path_to_save=path_to_save)
