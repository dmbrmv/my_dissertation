# Necessary columns to work with HydroATLAS
monthes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
lc_classes = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
]
wetland_classes = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
hydrology_variables = [
    item
    for sublist in [
        ["inu_pc_ult"],
        ["lka_pc_use"],
        ["lkv_mc_usu"],
        ["rev_mc_usu"],
        ["dor_pc_pva"],
        ["gwt_cm_sav"],
    ]
    for item in sublist
]
physiography_variables = [
    item for sublist in [["ele_mt_sav"], ["slp_dg_sav"], ["sgr_dk_sav"]] for item in sublist
]
climate_variables = [
    item
    for sublist in [
        ["clz_cl_smj"],
        ["cls_cl_smj"],
        ["tmp_dc_s{}".format(i) for i in monthes],
        ["pre_mm_s{}".format(i) for i in monthes],
        ["pet_mm_s{}".format(i) for i in monthes],
        ["aet_mm_s{}".format(i) for i in monthes],
        ["ari_ix_sav"],
        ["cmi_ix_s{}".format(i) for i in monthes],
        ["snw_pc_s{}".format(i) for i in monthes],
    ]
    for item in sublist
]
landcover_variables = [
    item
    for sublist in [
        ["glc_cl_smj"],
        ["glc_pc_s{}".format(i) for i in lc_classes],
        ["wet_pc_s{}".format(i) for i in wetland_classes],
        ["for_pc_sse"],
        ["crp_pc_sse"],
        ["pst_pc_sse"],
        ["ire_pc_sse"],
        ["gla_pc_sse"],
        ["prm_pc_sse"],
    ]
    for item in sublist
]
soil_and_geo_variables = [
    item
    for sublist in [
        ["cly_pc_sav"],
        ["slt_pc_sav"],
        ["snd_pc_sav"],
        ["soc_th_sav"],
        ["swc_pc_syr"],
        ["swc_pc_s{}".format(i) for i in monthes],
        ["kar_pc_sse"],
        ["ero_kh_sav"],
    ]
    for item in sublist
]
urban_variables = [
    item for sublist in [["urb_pc_sse"], ["hft_ix_s93"], ["hft_ix_s09"]] for item in sublist
]
