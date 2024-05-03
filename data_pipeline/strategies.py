from abc import ABC, abstractmethod
from typing import List, Dict
import logging
from datetime import datetime as dt, timedelta as td

import polars as pl #type: ignore
import numpy as np #type: ignore

import data_pipeline._functions_ as Fns

#StockDataset is a wrapper around a polars dataframe or any other framework's like pandas, 
#it is used to keep track of the dataset name and the data itself
class StockDataset:
    def __init__(self, data: dict = {}, name: str = None) -> None:
        self.data = data
        self.name = name
        self.splits = []
        self.columns = []
        self.shape = []
        self.steps = []

#Extraction strategies are standalone ways to load data depending on the source, they are independent and mutually exclusive
#Loading Data
class DataStrategy(ABC):
    @abstractmethod   
    def apply(self, dataset: StockDataset = None) -> StockDataset:
        #for specific loaders implement efficient column selection logic
        pass

class GCSCSVLoadStrategy(DataStrategy):
    def __init__(self, bucket: str, use_columns: List[str] = [None], dataset_name: str = 'StockHistory') -> None:
        self.bucket = bucket
        self.use_columns = use_columns
        self.dataset_name = dataset_name
        self.description = 'Load CSV from Google Cloud Storage using polars'

    def apply(self, dataset: StockDataset = None) -> StockDataset:
        data = {}
        if self.use_columns == [None]:
            data['full_df'] = pl.read_csv(self.bucket)
        else:
            data['full_df'] = pl.read_csv(self.bucket)[self.use_columns]

        dataset = StockDataset(data = data, name = self.dataset_name)
        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])
        return dataset
            
class LocalCSVLoadStrategy(DataStrategy):
    def __init__(self, path: str, use_columns: List[str] = [None], dataset_name: str = 'StockHistory') -> None:
        self.path = path
        self.use_columns = use_columns
        self.dataset_name = dataset_name
        self.description = 'Load CSV from local file using polars'

    def apply(self, dataset: StockDataset = None) -> StockDataset:
        data = {}
        if self.use_columns == [None]:
            data['full_df'] = pl.read_csv(self.path)

        else:
            data['full_df'] = pl.read_csv(self.path)[self.use_columns]

        dataset = StockDataset(data = data, name = self.dataset_name)
        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset

    
#Preprocessing strategies are strategies that only apply to our use case and our dataset. They can be applied before splitting the data
#into training, validation, and test sets. They are not mutually exclusive and can be applied in almost any order.
class DataCastingStrategy(DataStrategy):
    """Casting for polars dataset
    """
    def __init__(self, dtypes: Dict[str, pl.DataType]) -> None:
        self.dtypes = dtypes
        self.description = 'Casting data types of a polars dataset'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            dataset.data[df_name] = df.with_columns([pl.col(name).cast(dtype) for name, dtype in list(self.dtypes.items())])

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset
    
class FillInformationStrategy(DataStrategy):
    def __init__(self, date_column: str = 'Date', id_column: str = 'Ticker', interval: str = '1d') -> None:
        self.date_column = date_column
        self.id_column = id_column
        self.interval = interval
        self.description = 'Forward-Fill missing information in a polars dataset with date column ordering for each ticker'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            dfs = []
            for idx in df[self.id_column].unique().to_list():
                idx_df = df.filter(pl.col(self.id_column) == idx)
                idx_df = idx_df.sort(by = 'Date') #oldest to newest
                start_date = idx_df[self.date_column].min()
                end_date = idx_df[self.date_column].max()
                date_range = pl.DataFrame(data = pl.date_range(start_date, end_date, interval=self.interval, eager=True).alias(self.date_column))
                idx_df = date_range.join(idx_df, on=self.date_column, how='left')
                idx_df = idx_df.fill_null(strategy='forward')
                dfs.append(idx_df)
                dataset.data[df_name] = pl.concat(dfs)

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset
    
class TechnicalIndicatorsStrategy(DataStrategy):
    @classmethod
    def get_indicator_names(cls):
        indicators = ['Close_weighted_moving_avg_ratios',
                  'RSI', 
                  'bollinger_interval_ratio',
                  'stochastic_oscillator',
                  'atr',
                  'adx',
                  'CCI',
                  'tenkan_kijun',
                  'senkou_cloud',
                  'chinkou']
        return indicators
    
    def __init__(self, id_column: str = 'Ticker') -> None:
        self.id_column = id_column
        self.description = 'Create technical indicators for each ticker'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            dfs = []
            for idx in df[self.id_column].unique().to_list():
                idx_df = df.filter(pl.col(self.id_column) == idx)
                idx_df = idx_df.sort(by = 'Date') #oldest to newest

                #Simple Moving Average
                idx_df = Fns.moving_averages(idx_df)

                #Relative Strength Index
                idx_df = Fns.rsi(idx_df)

                #Bollinger Bands
                idx_df = Fns.bollinger_indicator(idx_df)

                #Stochastic oscillator
                idx_df = Fns.stochastic_oscillator(idx_df)

                #ADX
                idx_df = Fns.adx(idx_df, period = 15)

                #CCI
                idx_df = Fns.cci(idx_df, period = 15)

                #Ichimoku cloud
                idx_df = Fns.ichimoku_cloud(idx_df)

                dfs.append(idx_df)

            dataset.data[df_name] = pl.concat(dfs)

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])
        return dataset
    
#Processing strategies are strategies that only apply to our use case and our dataset. They should only be applied before splitting the data
#and at a row or ticker level. They are not mutually exclusive and should be applied in order of appearance as follows.

class RowLevelEnrichmentStrategy(DataStrategy):
    def __init__(self) -> None:
        self.description = 'Enriching the dataset with row-level features'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            df = df.with_columns([
                (pl.col('High') - pl.col('Low')).alias('1_d_variation'),
                (pl.col('Close') - pl.col('Open')).alias('1_d_return'),
                (pl.col('Close')*pl.col('Volume')).alias('eod_value_proxy'),
                ((pl.col('High') - pl.col('Low'))*pl.col('Volume')).alias('variation_strength'),
                ((pl.col('Close') - pl.col('Open'))*pl.col('Volume')).alias('return_strength')
                ])
            dataset.data[df_name] = df

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])
        return dataset
    
class TargetVariableCreationStrategy(DataStrategy):
    def __init__(self, target_column: str = 'Close', id_column:str = 'Ticker', horizon: int = 30) -> None:
        self.target_column = target_column
        self.horizon = horizon
        self.id_column = id_column
        self.description = 'Creating target variable for each ticker'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            dfs = []
            for idx in df[self.id_column].unique().to_list():
                idx_df = df.filter(pl.col(self.id_column) == idx)
                idx_df = idx_df.sort(by = 'Date') #oldest to newest

                idx_df = idx_df.with_columns(
                    (pl.col(self.target_column).shift(-self.horizon)/pl.col(self.target_column)).alias('target')
                    )

                dfs.append(idx_df)
            dataset.data[df_name] = pl.concat(dfs)

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])
        return dataset
    
class FeatureEngineeringStrategy(DataStrategy):
    def __init__(self,
                 timepoints: List[int] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
                 id_column:str = 'Ticker') -> None:
        
        self.timepoints = timepoints
        self.id_column = id_column
        self.description = 'Creating features for each ticker'

    def apply(self, dataset: StockDataset) -> StockDataset:

        #output of enrichment
        cols_of_interest = ['Close', 
                            'Volume',
                            '1_d_variation',
                            '1_d_return',
                            'eod_value_proxy',
                            'variation_strength',
                            'return_strength']
        
        new_features = [( pl.col(col)/(pl.col(col).shift(i)) ).alias(f'{col}_{i}_d_ratio') for col in cols_of_interest for i in self.timepoints]
        
        for df_name, df in dataset.data.items():
            dfs = []
            for idx in df[self.id_column].unique().to_list():
                idx_df = df.filter(pl.col(self.id_column) == idx)
                idx_df = idx_df.sort(by = 'Date')
                idx_df = idx_df.with_columns(new_features)
                #include column in features if it is a ratio (i.e. _d_ratio is in the name)
                self.features = [_ for _ in idx_df.columns if '_d_ratio' in _]

                dfs.append(idx_df)
                
            df = pl.concat(dfs)
            del dfs
            
            #Get weighted average of features through the timepoints, where the most recent data has the highest weight
            reversed_timepoints = self.timepoints[::-1]
            weights = np.array(reversed_timepoints)/sum(reversed_timepoints)

            #for each feature, create a column that is the weighted average of the feature through the timepoints
            idx = [i for i in range(len(self.timepoints))]
            for col in cols_of_interest:
                weighted_product = [pl.col(f'{col}_{self.timepoints[i]}_d_ratio')*weights[i] for i in idx]

                df = df.with_columns(pl.sum_horizontal(weighted_product).alias(f'{col}_weighted_avg'))
                df = df.drop([f'{col}_{self.timepoints[i]}_d_ratio' for i in idx])

            self.features = [_ for _ in df.columns if '_weighted_avg' in _]

            features_from_past_step = TechnicalIndicatorsStrategy.get_indicator_names()

            final_columns = ['Date', 'Ticker', 'target'] + self.features + features_from_past_step
            dataset.data[df_name] = df.select(final_columns)

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset

class TimeFeaturesStrategy(DataStrategy):
    def __init__(self, date_column: str = 'Date') -> None:
        self.date_column = date_column
        self.description = 'Creating time features for each row'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            df = df.with_columns([
                pl.col(self.date_column).dt.year().alias('year'),
                pl.col(self.date_column).dt.month().alias('month'),
                pl.col(self.date_column).dt.day().alias('day'),
                pl.col(self.date_column).dt.week().alias('week'),
                pl.col(self.date_column).dt.weekday().alias('weekday')
                ])
            dataset.data[df_name] = df
        
        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])
        return dataset

    
class DropTargetNullsStrategy(DataStrategy):
    def __init__(self, target_column: str = 'target') -> None:
        self.target_column = target_column
        self.description = 'Dropping rows with null target values'

    def apply(self, dataset: StockDataset) -> StockDataset:
        for df_name, df in dataset.data.items():
            dataset.data[df_name] = df.filter(pl.col(self.target_column).is_not_null())

        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset

#After we apply the split we have to be carefulll, as we are no longer dealing with a dataframe but a dictionary of dataframes. After this,
#any transformation should be calibrated (if needed) on the train data and applied to the validation and test data afterwards.
class DataSplittingStrategy(DataStrategy):
    def __init__(self, train_min_date: dt, train_max_date: dt, validation_max_date: dt, calibration_max_date: dt) -> None:
        self.train_min_date = train_min_date
        self.train_max_date = train_max_date
        self.validation_max_date = validation_max_date
        self.calibration_max_date = calibration_max_date
        self.description = 'Splitting data into training, validation, (possibly calibration) and test sets'

    def apply(self, dataset: StockDataset) -> Dict[str, StockDataset]:
        data = {}

        if 'full_df' not in dataset.data:
            raise ValueError('No data to split')
        

        df = dataset.data['full_df']
        #training subset
        train_filter = (pl.col('Date') <= self.train_max_date) & (pl.col('Date') >= self.train_min_date)
        data['train'] = df.filter(train_filter)

        #validation subset
        val_filter = (pl.col('Date') > self.train_max_date) & (pl.col('Date') <= self.validation_max_date)
        data['validation'] = df.filter(val_filter)

        #calibration subset
        if self.calibration_max_date is None:
            #data['calibration'] = None
            test_filter = pl.col('Date') > self.validation_max_date
            self.description = 'Splitting data into training, validation and test sets'

        else:
            self.description = 'Splitting data into training, validation, calibration and test sets'
            calib_filter = (pl.col('Date') > self.validation_max_date) & (pl.col('Date') <= self.calibration_max_date)
            data['calibration'] = df.filter(calib_filter)
            test_filter = pl.col('Date') > self.calibration_max_date

        #test subset
        data['test'] = df.filter(test_filter)

        dataset.data = data
        dataset.splits = [_ for _ in list(data.keys()) if _ is not None]
        dataset.steps.append(self.description)
        dataset.shape.append([(df_name, df.shape) for df_name, df in dataset.data.items()])
        dataset.columns.append([(df_name, df.columns) for df_name, df in dataset.data.items()])

        return dataset