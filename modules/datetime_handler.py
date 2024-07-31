import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Union
from modules.helpers.validators import ColumnTypeValidators

class DatetimeHandler:
    
    def __init__(self) -> None:
        """"""
        pass

    def convert_to_datetime(self, dataframe: pd.DataFrame, column: Union[str, int], format: str = '%Y-%m-%d %H:%M:%S'):
        df_copy = dataframe.copy()
        df_copy[column] = pd.to_datetime(dataframe[column], format=format, errors='coerce')
        return df_copy

    @ColumnTypeValidators.is_column_exists
    def correct_invalid_datetime(self, dataframe: pd.DataFrame, column: Union[str, int], format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
        '''Correct invalid datetime entries in the specified column by attempting to parse them into a valid datetime format.'''
        def correct_date(date_str):
            try:
                return pd.to_datetime(date_str, format=format, errors='coerce')
            except:
                return pd.NaT

        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column].apply(correct_date)
        return df_copy
    
    @ColumnTypeValidators.datetime_required
    def extract_components(self, dataframe: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
        '''Extract date components (year, month, day, hour, minute, second) into separate columns.'''
        
        df_copy = dataframe.copy()
        df_copy['year'] = dataframe.iloc[:, column].dt.year
        df_copy['month'] = dataframe.iloc[:, column].dt.month
        df_copy['day'] = dataframe.iloc[:, column].dt.day
        df_copy['hour'] = dataframe.iloc[:, column].dt.hour
        df_copy['minute'] = dataframe.iloc[:, column].dt.minute
        df_copy['second'] = dataframe.iloc[:, column].dt.second
        return df_copy

    @ColumnTypeValidators.datetime_required
    def reformat_date(self, dataframe: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
        '''Reformat the datetime objects in the specified column to a different string format.'''
        
        df_copy = dataframe.copy()
        df_copy.iloc[:, column] = df_copy.iloc[:, column].apply(lambda x: x.strftime('%d-%m-%Y %H:%M:%S') if pd.notnull(x) else x)
        return df_copy

    @ColumnTypeValidators.datetime_required
    def calculate_datetime_differences(self, dataframe: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
        '''Calculate the difference between consecutive datetime entries in the specified column.'''
        df_copy = dataframe.copy()
        df_copy['time_diff'] = df_copy.iloc[:, column].diff().dt.total_seconds()
        return df_copy

    @ColumnTypeValidators.datetime_required
    def convert_datetime_to_different_timezones(self, dataframe: pd.DataFrame, column: Union[str, int], from_tz='UTC', to_tz='America/New_York') -> pd.DataFrame:
        '''Convert datetime objects from one timezone to another.'''
        
        df_copy = dataframe.copy()
        from_zone = pytz.timezone(from_tz)
        to_zone = pytz.timezone(to_tz)
        def convert_timezone(dt):
            if pd.notnull(dt):
                dt = from_zone.localize(dt) if dt.tzinfo is None else dt
                return dt.astimezone(to_zone)
            return dt
        df_copy.iloc[:, column] = df_copy.iloc[:, column].apply(convert_timezone)
        return df_copy

    @ColumnTypeValidators.datetime_required
    def shift_time(self, dataframe: pd.DataFrame, column: Union[str, int], shift_value=1, unit='days') -> pd.DataFrame:
        '''Shift the datetime values in the specified column by a given amount.'''
        
        df_copy = dataframe.copy()
        df_copy[column] = df_copy[column] + pd.to_timedelta(shift_value, unit=unit)
        return df_copy
    
    
    @ColumnTypeValidators.datetime_required
    def moving_average(self, dataframe: pd.DataFrame, column: Union[str, int], window: int = 3) -> pd.DataFrame:
        '''Calculate the moving average of a datetime series over a specified window.'''
        df_copy = dataframe.copy()
        df_copy[f'{column}_moving_avg'] = df_copy[column].rolling(window=window).mean()
        return df_copy
    
    @ColumnTypeValidators.datetime_required
    def exponential_smoothing(self, dataframe: pd.DataFrame, column: Union[str, int], alpha: float = 0.5) -> pd.DataFrame:
        '''Apply exponential smoothing to a datetime series.'''
        df_copy = dataframe.copy()
        df_copy[f'{column}_exp_smooth'] = df_copy[column].ewm(alpha=alpha).mean()
        return df_copy
