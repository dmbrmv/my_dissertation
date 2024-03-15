import pandas as pd
from spotpy.objectivefunctions import rmse
from spotpy.parameter import Uniform

from model_scripts import hbv


class hbv_setup(object):
    """
    16 parameters
    BETA   - parameter that determines the relative contribution to runoff from rain or snowmelt
             [1, 6]
    CET    - Evaporation correction factor
             (should be 0 if we don't want to change (Oudin et al., 2005) formula values)
             [0, 0.3]
    FC     - maximum soil moisture storage
             [50, 500]
    K0     - recession coefficient for surface soil box (upper part of SUZ)
             [0.01, 0.4]
    K1     - recession coefficient for upper groudwater box (main part of SUZ)
             [0.01, 0.4]
    K2     - recession coefficient for lower groudwater box (whole SLZ)
             [0.001, 0.15]
    LP     - Threshold for reduction of evaporation (SM/FC)
             [0.3, 1]
    MAXBAS - routing parameter, order of Butterworth filter
             [1, 7]
    PERC   - percolation from soil to upper groundwater box
             [0, 3]
    UZL    - threshold parameter for grondwater boxes runoff (mm)
             [0, 500]
    PCORR  - Precipitation (input sum) correction factor
             [0.5, 2]
    TT     - Temperature which separate rain and snow fraction of precipitation
             [-1.5, 2.5]
    CFMAX  - Snow melting rate (mm/day per Celsius degree)
             [1, 10]
    SFCF   - SnowFall Correction Factor
             [0.4, 1]
    CFR    - Refreezing coefficient
             [0, 0.1] (usually 0.05)
    CWH    - Fraction (portion) of meltwater and rainfall which retain in snowpack (water holding capacity)
             [0, 0.2] (usually 0.1)

    """

    beta = Uniform(low=1.0, high=6.0, optguess=3.0)
    cet = Uniform(low=0.0, high=0.3, optguess=0.15)
    fc = Uniform(low=50.0, high=500.0, optguess=150.0)
    k0 = Uniform(low=0.01, high=0.4, optguess=0.2)
    k1 = Uniform(low=0.01, high=0.4, optguess=0.2)
    k2 = Uniform(low=0.001, high=0.15, optguess=0.1)
    lp = Uniform(low=0.3, high=1.0, optguess=0.99)
    maxbas = Uniform(low=1.0, high=7.0, optguess=3.0)
    perc = Uniform(low=0.0, high=3.0, optguess=2.0)
    uzl = Uniform(low=0.0, high=500.0, optguess=150.0)
    pcorr = Uniform(low=0.5, high=2.0, optguess=2.0)
    tt = Uniform(low=-1.5, high=2.5, optguess=1.0)
    cfmax = Uniform(low=1.0, high=10.0, optguess=5.0)
    sfcf = Uniform(low=0.4, high=1.0, optguess=0.8)
    cfr = Uniform(low=0.0, high=1.0, optguess=0.05)
    cwh = Uniform(low=0.0, high=0.2, optguess=0.1)

    def __init__(self, data_file: pd.DataFrame, obj_func=None):
        self.obj_func = obj_func
        self.obs_q = data_file.loc[:, "q_mm_day"].values
        temp_max = data_file.loc[:, "t_min_e5l"].values
        temp_min = data_file.loc[:, "t_max_e5l"].values
        temp_mean = (temp_max + temp_min) / 2
        evap = data_file.loc[:, "Ep"].values
        prcp = data_file.loc[:, "prcp_e5l"].values

        self.model_df = pd.DataFrame()
        self.model_df["Temp"] = temp_mean
        self.model_df["Evap"] = evap
        self.model_df["Prec"] = prcp
        self.model_df["Q_mm"] = self.obs_q
        self.model_df.index = data_file.index

    def simulation(self, params):
        sim_results = hbv.simulation(self.model_df, params)

        return sim_results

    def evaluation(self):
        return self.obs_q

    def objectivefunction(
        self, simulation: object, evaluation: object, params: object = None
    ) -> object:
        """
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run

        Returns:
            object:
        """

        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like
