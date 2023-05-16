import glob
import pandas as pd
from scripts.data_readers import read_gauge, get_params
from pathlib import Path

calibration_place = Path('./cal_res')
hbv_calibration = Path(f'{calibration_place}/hbv')
hbv_calibration.mkdir(exist_ok=True, parents=True)
gr4j_calibration = Path(f'{calibration_place}/gr4j')
gr4j_calibration.mkdir(exist_ok=True, parents=True)

really_bad_gauges = ['48011', '72004', '12614', '12305', '4036', '4226',
                     '5190', '10357', '80174', '10549', '9344', '10688',
                     '84448', '74425', '78507', '84803', '8256', '8359',
                     '5607', '84108', '72246', '8139', '10539', '75527',
                     '4197', '78343', '75428', '84200', '19383', '78225',
                     '84213', '19135', '12354', '9331', '72518', '4068',
                     '84398', '2164', '5613', '77362', '8120', '4012',
                     '5352', '84043', '84453', '8281', '48008', '83260',
                     '78501', '84119', '76145', '10058', '84245', '4235',
                     '84295', '75780', '72450', '5199', '9309', '48056',
                     '75110', '4240', '77164', '84019', '4005', '5611',
                     '84157', '842000', '49128', '9546', '5604', '76325',
                     '9511', '10060', '49095', '84315', '72198', '77327',
                     '78620', '78163', '84185', '76644', '9345', '7171',
                     '4071', '84400', '19111', '19008', '6563', '75271',
                     '76470', '70129', '72744', '9355', '11327', '70102',
                     '11374', '75222', '19167', '75387', '78261',
                     '10584', '78499', '84354', '10548', '11675']
hbv_df = pd.DataFrame()
gr4j_df = pd.DataFrame()
for i, gauge in enumerate([i.split('/')[-1][:-3]
                           for i in
                           glob.glob('../geo_data/great_db/nc_all_q/*.nc')]):

    if gauge in really_bad_gauges:
        pass
    for model in ['hbv', 'gr4j']:
        if model == 'hbv':
            train, test = read_gauge(gauge_id=gauge)
            hbv_df.loc[i, 'gauge_id'] = gauge
            hbv_df.loc[i, 'NSE'] = get_params(gauge_id=gauge,
                                              model_name=model,
                                              params_path=hbv_calibration,
                                              train=train, test=test,
                                              with_plot=True)
        elif model == 'gr4j':
            train, test = read_gauge(gauge_id=gauge)
            gr4j_df.loc[i, 'gauge_id'] = gauge
            gr4j_df.loc[i, 'NSE'] = get_params(gauge_id=gauge,
                                               model_name=model,
                                               params_path=gr4j_calibration,
                                               train=train, test=test,
                                               with_plot=True)
hbv_df.to_csv('./res_hbv.csv', index=False)
gr4j_df.to_csv('./res_gr4j.csv', index=False)
