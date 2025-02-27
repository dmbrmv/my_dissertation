import geopandas as gpd
from scripts.feature_selection import featureXtractor, save_results
from tqdm import tqdm

# set path to shape file
home_folder = "../data"
# path to geodataframe
shape_file_path = f"{home_folder}/great_db/geometry"
# set path to downloaded HydroATLAS
gdb_file_path = f"{home_folder}/hydro_atlas/BasinATLAS_v10.gdb"
# set path where results will be stored
path_to_save = f"{home_folder}/static_attributes/don_10_03"

# Read shape file with geometry column
my_shape = gpd.read_file(
    "/home/anton/dima_experiments/Vip_gZ/data/great_db/don_03_10/res/don_gauges_03_10_2023_ws.gpkg"
)

output = list()
for ws in tqdm(my_shape.loc[:, "geometry"]):
    output.append(featureXtractor(user_ws=ws, gdb_file_path=gdb_file_path))

save_results(extracted_data=output, gauge_ids=my_shape.loc[:, "gauge_id"], path_to_save=path_to_save)
