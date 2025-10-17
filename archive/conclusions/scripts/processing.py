import pandas as pd


def split_by_hydro_year(discharge_obs: pd.Series):
    return {
        str(year): discharge_obs[f"10/01/{year}" : f"10/01/{year+1}"]
        for year in discharge_obs.index.year.unique()
    }


def split_by_year(discharge_obs: pd.Series):
    return {
        str(year): discharge_obs[f"01/01/{year}" : f"12/31/{year}"]
        for year in discharge_obs.index.year.unique()
    }


def read_table_gauge_str(
    table_path: str, index_filter: pd.Index = pd.Index([]), filter_zero: bool = False
) -> tuple[pd.DataFrame, float, float, float]:
    table = pd.read_csv(table_path)
    if "Unnamed: 0" in table.columns:
        table = table.rename(columns={"Unnamed: 0": "gauge_id"})
    table["gauge_id"] = table["gauge_id"].astype(str)
    table = table.set_index("gauge_id")

    if index_filter.empty:
        pass
    else:
        table = table.loc[table.index.isin(index_filter)]

    if filter_zero:
        table = table[(table["precision"] != 0) & (table["recall"] != 0)]
    median_recall, median_precision, median_rmse = (
        table["recall"].median(),
        table["precision"].median(),
        table["rmse"].median(),
    )
    return table, median_recall, median_precision, median_rmse
