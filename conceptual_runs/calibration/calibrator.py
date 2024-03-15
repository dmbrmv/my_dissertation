from spotpy.objectivefunctions import rmse
import spotpy
from calibration.model_setups import hbv_setup

models_dict = {"hbv": hbv_setup, "gr4j": gr4j_setup}


def calibrate_gauge(
    df, res_calibrate: str, hydro_models=["hbv", "gr4j"], iterations=600
):
    for model in hydro_models:
        print(f"\n--------------\nCurrent model {model}\n--------------\n")
        spotpy_setup = models_dict[model](df, obj_func=rmse)

        sampler = spotpy.algorithms.sceua(
            spotpy_setup,
            # result file
            dbname=f"{res_calibrate}",
            dbformat="csv",
            save_sim=False,
        )
        # if test:
        # 	return sampler, spotpy_setup
        sampler.sample(iterations)
        # Блок вытаскивания идеальных лучших параметров
        results = sampler.getdata()
        best_params = spotpy.analyser.get_best_parameterset(results, maximize=False)
        bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)

        # res parameters
        with open(f"{res_calibrate}", "a") as f:
            f.write(
                f"""{model}: \n
RMSE: {bestobjf}\nparams: {best_params}\n\n{'-----' * 5}\n\n"""
            )
