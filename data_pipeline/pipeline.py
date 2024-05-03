from data_pipeline.strategies import *
from datetime import datetime as dt, timedelta as td

class Pipeline:
    def __init__(self) -> None:
        self.strategies = []

    def add_strategy(self, strategy: DataStrategy) -> None:
        self.strategies.append(strategy)

    def apply(self, dataset: StockDataset = None) -> None:
        for strategy in self.strategies:
            dataset = strategy.apply(dataset)
        return dataset
    
class DevelopmentPipeline:
    def __init__(self, input_data_path: str, data_source: str = 'bucket') -> None:
        self.input_data_path = input_data_path
        self.data_source = data_source
        self.pipeline = None
    
    def build(self) -> None:
        data_source = self.data_source
        input_data_path = self.input_data_path

        if data_source == 'bucket':
            #Load data from GCS bucket
            step1 = GCSCSVLoadStrategy(bucket = input_data_path, use_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
        else:
            step1 = LocalCSVLoadStrategy(path = input_data_path, use_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
        #get the right data types
        dtypes = {
            'Date': pl.Date(),
            'Open': pl.Float32(),
            'High': pl.Float32(),
            'Low': pl.Float32(),
            'Close': pl.Float32(),
            'Volume': pl.Int64(),
            }
        step2 = DataCastingStrategy(dtypes = dtypes)

        #Fill information to have a complete dataset for every day of the calendar
        step3 = FillInformationStrategy(date_column = 'Date', id_column = 'Ticker', interval = '1d')

        #Techincal indicators
        step4 = TechnicalIndicatorsStrategy(id_column = 'Ticker')

        #Enrich data
        step5 = RowLevelEnrichmentStrategy()

        #Create target variable
        step6 = TargetVariableCreationStrategy(target_column = 'Close', id_column = 'Ticker', horizon = 15)

        #Feature engineering
        timepoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        step7 = FeatureEngineeringStrategy(id_column = 'Ticker', timepoints = timepoints)

        #Get date features
        step8 = TimeFeaturesStrategy(date_column = 'Date')

        #drop if any missing values on target column
        step9 = DropTargetNullsStrategy(target_column = 'target')

        #Finally, split the data by time
        #We will use up to 3 years ago for training, up to 2 years ago for validation and up to 1 year ago for calibration
        train_max_date = (dt.today() - td(days=365*3))
        train_min_date = train_max_date - td(days=365*10)
        validation_max_date = (dt.today() - td(days=365*2))
        calibration_max_date = (dt.today() - td(days=365*1))

        step10 = DataSplittingStrategy(train_min_date = train_min_date,
                                    train_max_date = train_max_date,
                                    validation_max_date = validation_max_date,
                                    calibration_max_date = calibration_max_date)

        pipeline = Pipeline()
        pipeline.add_strategy(step1)
        pipeline.add_strategy(step2)
        pipeline.add_strategy(step3)
        pipeline.add_strategy(step4)
        pipeline.add_strategy(step5)
        pipeline.add_strategy(step6)
        pipeline.add_strategy(step7)
        pipeline.add_strategy(step8)
        pipeline.add_strategy(step9)
        pipeline.add_strategy(step10)

        self.pipeline = pipeline

    def run(self) -> StockDataset:

        if self.pipeline is None:
            self.build()
        
        dataset = self.pipeline.apply()
        return dataset
    
class TrainingPipeline:
    def __init__(self, input_data_path: str, data_source: str = 'bucket') -> None:
        self.input_data_path = input_data_path
        self.data_source = data_source
        self.pipeline = None
    
    def build(self) -> None:
        data_source = self.data_source
        input_data_path = self.input_data_path

        if data_source == 'bucket':
            #Load data from GCS bucket
            step1 = GCSCSVLoadStrategy(bucket = input_data_path, use_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
        else:
            step1 = LocalCSVLoadStrategy(path = input_data_path, use_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
        #get the right data types
        dtypes = {
            'Date': pl.Date(),
            'Open': pl.Float32(),
            'High': pl.Float32(),
            'Low': pl.Float32(),
            'Close': pl.Float32(),
            'Volume': pl.Int64(),
            }
        step2 = DataCastingStrategy(dtypes = dtypes)

        #Fill information to have a complete dataset for every day of the calendar
        step3 = FillInformationStrategy(date_column = 'Date', id_column = 'Ticker', interval = '1d')

        #Techincal indicators
        step4 = TechnicalIndicatorsStrategy(id_column = 'Ticker')

        #Enrich data
        step5 = RowLevelEnrichmentStrategy()

        #Create target variable
        step6 = TargetVariableCreationStrategy(target_column = 'Close', id_column = 'Ticker', horizon = 15)

        #Feature engineering
        timepoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        step7 = FeatureEngineeringStrategy(id_column = 'Ticker', timepoints = timepoints)

        #Get date features
        step8 = TimeFeaturesStrategy(date_column = 'Date')

        #drop if any missing values on target column
        step9 = DropTargetNullsStrategy(target_column = 'target')

        #Finally, split the data by time
        #We will use up to 6 months ago for training, up to 3 months ago for validation (this will be calibration)
        # and the last 3 months for testing
        train_max_date = (dt.today() - td(days=180))
        train_min_date = train_max_date - td(days=365*10)
        validation_max_date = (dt.today() - td(days=90))
        calibration_max_date = None

        step10 = DataSplittingStrategy(train_min_date = train_min_date,
                                    train_max_date = train_max_date,
                                    validation_max_date = validation_max_date,
                                    calibration_max_date = calibration_max_date)

        pipeline = Pipeline()
        pipeline.add_strategy(step1)
        pipeline.add_strategy(step2)
        pipeline.add_strategy(step3)
        pipeline.add_strategy(step4)
        pipeline.add_strategy(step5)
        pipeline.add_strategy(step6)
        pipeline.add_strategy(step7)
        pipeline.add_strategy(step8)
        pipeline.add_strategy(step9)
        pipeline.add_strategy(step10)

        self.pipeline = pipeline

    def run(self) -> StockDataset:

        if self.pipeline is None:
            self.build()
        
        dataset = self.pipeline.apply()
        return dataset

    
class BatchProductionPipeline:
    pass

class RealTimeProductionPipeline:
    pass
