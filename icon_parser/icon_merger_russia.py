from scripts.icon_to_ws import Icon_merger
from datetime import datetime
import gc

day = datetime.today().strftime('%Y-%m-%d')
h_m = datetime.today().strftime('%H-%M')
print(f'It is time to merge icon for {day} at {h_m}')
icon_merger = Icon_merger(
    ws_path='../geo_data/great_db/geometry/russia_ws.gpkg',
    icon_gauges='../geo_data/icon_oct_23/',
    place_to_save='../geo_data/icon_russia/',
    dataset_name='icon_7_days')
icon_merger.merger()
# clear garbage
gc.collect()
