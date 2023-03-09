import pandas as pd


def read_with_date_index(file_path: str):
    
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    return data
