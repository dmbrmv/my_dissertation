from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
import os

from hydro_models import hbv, gr4j_cema_neige


class gr4j_setup(object):
    '''
        Input:
    1. Meteorological forcing
        data - pandas dataframe with correspondent columns
        for temperature ('Temp') and potential evaporation ('Evap')
    2. list of model parameters:
        GR4J params:
        X1 : production store capacity (X1 - PROD) [mm]
            [0, 1500]
        X2 : intercatchment exchange coefficient (X2 - CES) [mm/day]
            [-10, 5]
        X3 : routing store capacity (X3 - ROUT) [mm]
            [1, 500]
        X4 : time constant of unit hydrograph (X4 - TB) [day]
            [0.5, 4.0]
        Cema-Neige snow model parameters:
        X5 (CTG): dimensionless weighting coefficient
        of the snow pack thermal state
            [0, 1]
        X6 (Kf): day-degree rate of melting (mm/(day*celsium degree))
            [1, 10]
        X7 (TT) : temperature which separates rain and
        snow fraction of precipitation
            [-1.5, 2.5]
    '''
    x1 = Uniform(low=0., high=3000.)
    x2 = Uniform(low=-10., high=10.)
    x3 = Uniform(low=0., high=1000.)
    x4 = Uniform(low=0, high=20)
    x5 = Uniform(low=0., high=1., optguess=0.0)
    x6 = Uniform(low=1., high=10., optguess=1.9)
    x7 = Uniform(low=-1.5, high=2.5, optguess=0.55)

    def __init__(self, data, obj_func=None):
        self.Name = 'GR4J'
        # Find Path to Hymod on users system
        self.owd = os.path.dirname(os.path.realpath(__file__))
        self.hymod_path = self.owd

        self.obj_func = obj_func
        self.date = list(data.index)
        self.trueObs = list(data["Q_mm"])
        self.Temp = list(data['Temp'])
        self.Evap = list(data['Evap'])
        self.Prec = list(data['Prec'])
        self.full_df = data

    def whatIsMyName(self):
        return self.Name

    def simulation(self, x):
        data = self.full_df
        # Here the model is actualy started with a unique parameter combination
        # that it gets from spotpy for each time the model is called
        sim = gr4j_cema_neige.simulation(data, x)
        return sim[366:]

    def evaluation(self):
        # The first year of simulation data is ignored (warm-up)
        return self.trueObs[366:]

    def return_dates(self):
        # The first year of simulation data is ignored (warm-up)
        return self.date[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like


class hbv_setup(object):
    '''
      # 16 parameters
    # BETA   - parameter that determines the relative contribution to runoff
    # from rain or snowmelt
    #          [1, 6]
    # CET    - Evaporation correction factor
    #          (should be 0 if we don't want to change (Oudin et al., 2005)
    # formula values)
    #          [0, 0.3]
    # FC     - maximum soil moisture storage
    #          [50, 500]
    # K0     - recession coefficient for surface soil box
    # (upper part of SUZ)
    #          [0.01, 0.4]
    # K1     - recession coefficient for upper groudwater box
    # (main part of SUZ)
    #          [0.01, 0.4]
    # K2     - recession coefficient for lower groudwater box
    # (whole SLZ)
    #          [0.001, 0.15]
    # LP     - Threshold for reduction of evaporation (SM/FC)
    #          [0.3, 1]
    # MAXBAS - routing parameter, order of Butterworth filter
    #          [1, 7]
    # PERC   - percolation from soil to upper groundwater box
    #          [0, 3]
    # UZL    - threshold parameter for grondwater boxes runoff (mm)
    #          [0, 500]
    # PCORR  - Precipitation (input sum) correction factor
    #          [0.5, 2]
    # TT     - Temperature which separate rain and
    # snow fraction of precipitation
    #          [-1.5, 2.5]
    # CFMAX  - Snow melting rate (mm/day per Celsius degree)
    #          [1, 10]
    # SFCF   - SnowFall Correction Factor
    #          [0.4, 1]
    # CFR    - Refreezing coefficient
    #          [0, 0.1] (usually 0.05)
    # CWH    - Fraction (portion) of meltwater and rainfall which
    # retain in snowpack (water holding capacity)
    #          [0, 0.2] (usually 0.1)
    '''
    BETA = Uniform(low=1.0, high=6.0)
    CET = Uniform(low=0., high=0.3)
    FC = Uniform(low=50., high=700., optguess=50.)
    K0 = Uniform(low=0.05, high=0.99)
    K1 = Uniform(low=0.01, high=0.8)
    K2 = Uniform(low=0.001, high=0.15)
    LP = Uniform(low=0.3, high=1.)
    MAXBAS = Uniform(low=1., high=3.)
    PERC = Uniform(low=0., high=6.)
    UZL = Uniform(low=0., high=100.)
    PCORR = Uniform(low=0.5, high=2.)
    TT = Uniform(low=-2.5, high=2.5, optguess=0.55)
    CFMAX = Uniform(low=0.5, high=5., optguess=1.)
    SFCF = Uniform(low=1., high=1.5, optguess=1.)
    CFR = Uniform(low=0., high=0.01, optguess=0.005)
    CWH = Uniform(low=0., high=0.2, optguess=0.1)

    def __init__(self, data, obj_func=None):
        self.Name = 'HBV'
        # Find Path to Hymod on users system
        self.owd = os.path.dirname(os.path.realpath(__file__))
        self.hymod_path = self.owd

        self.obj_func = obj_func
        self.date = list(data.index)
        self.trueObs = list(data["Q_mm"])
        self.Temp = list(data['Temp'])
        self.Evap = list(data['Evap'])
        self.Prec = list(data['Prec'])
        self.full_df = data

    def whatIsMyName(self):
        return self.Name

    def simulation(self, x):
        data = self.full_df
        # Here the model is actualy started with a unique parameter combination
        # that it gets from spotpy for each time the model is called
        sim = hbv.simulation(data, x)
        # The first year of simulation data is ignored (warm-up)
        return sim[366:]

    def evaluation(self):
        # The first year of simulation data is ignored (warm-up)
        return self.trueObs[366:]

    def return_dates(self):
        # The first year of simulation data is ignored (warm-up)
        return self.date[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like
