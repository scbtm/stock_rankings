from metaflow import FlowSpec, step


class TuningFlow(FlowSpec):
    @step
    def start(self):
        """
        This step is used to get the data prepared for development. This step is the starting point of the flow. 
        """
        from data_pipeline.pipeline import DevelopmentPipeline

        import polars as pl
        import numpy as np
        import pandas as pd

        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

        from dotenv import load_dotenv
        import os

        # Load environment variables from .env
        load_dotenv()

        #Constants to be used throughout the step
        input_data_path = os.getenv("INPUT_DATA_PATH")
        data_source = os.getenv("DATA_SOURCE")
        target_threshold = 1.05

        #Get dataset object
        dataset = DevelopmentPipeline(input_data_path=input_data_path, data_source=data_source).run()

        #Get the data
        xtrain, ytrain = dataset.data['train'].drop(['target', 'Date', 'Ticker']), (dataset.data['train']['target'] >= target_threshold).cast(pl.Int64)
        xval, yval = dataset.data['validation'].drop(['target', 'Date', 'Ticker']), (dataset.data['validation']['target'] >= target_threshold).cast(pl.Int64)
        xcal, ycal = dataset.data['calibration'].drop(['target', 'Date', 'Ticker']), (dataset.data['calibration']['target'] >= target_threshold).cast(pl.Int64)
        xtest, ytest = dataset.data['test'].drop(['target', 'Date', 'Ticker']), (dataset.data['test']['target'] >= target_threshold).cast(pl.Int64)

        xtrain = xtrain.to_pandas()
        ytrain = ytrain.to_pandas().values

        xval = xval.to_pandas()
        yval = yval.to_pandas().values

        xcal = xcal.to_pandas()
        ycal = ycal.to_pandas().values

        xtest = xtest.to_pandas()
        ytest = ytest.to_pandas().values


        #train a dummy classifier to establish some basic baselines
        #Random classifier
        dummy = DummyClassifier(strategy='uniform')
        dummy.fit(xtrain, ytrain)
        dummy_preds = dummy.predict(xval)

        metrics = {}
        metrics['random_dummy'] = {'accuracy': accuracy_score(yval, dummy_preds),
                                   'precision': precision_score(yval, dummy_preds),
                                   'recall': recall_score(yval, dummy_preds),
                                   'f1': f1_score(yval, dummy_preds),
                                   'balanced_accuracy': balanced_accuracy_score(yval, dummy_preds)}
        

        #Most frequent classifier        
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(xtrain, ytrain)
        dummy_preds = dummy.predict(xval)

        metrics['most_frequent_dummy'] = {'accuracy': accuracy_score(yval, dummy_preds),
                                          'precision': precision_score(yval, dummy_preds),
                                          'recall': recall_score(yval, dummy_preds),
                                          'f1': f1_score(yval, dummy_preds),
                                          'balanced_accuracy': balanced_accuracy_score(yval, dummy_preds)}
        self.baseline_metrics = metrics

        dataset = {}
        dataset['xtrain'] = xtrain
        dataset['ytrain'] = ytrain

        dataset['xval'] = xval
        dataset['yval'] = yval

        dataset['xcal'] = xcal
        dataset['ycal'] = ycal

        dataset['xtest'] = xtest
        dataset['ytest'] = ytest

        self.dataset = dataset


        self.next(self.tune)


    @step
    def tune(self):
        """
        This step is used to tune the hyperparameters of the model. 
        """

        import optuna
        import catboost
        from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score


        import wandb

        from dotenv import load_dotenv
        import os

        # Load environment variables from .env
        load_dotenv()

        # Get W&B API key
        #wandb_api_key = os.getenv('WANDB_API_KEY')
        project_name = os.getenv('WANDB_PROJECT_NAME')
        entity = os.getenv('WANDB_ENTITY')

        # Initialize W&B
        wandb.init(project=project_name, entity=entity)

        def objective(trial, dataset):
            xtrain, ytrain = dataset['xtrain'], dataset['ytrain']
            xval, yval = dataset['xval'], dataset['yval']

            train_size = len(xtrain)
            #Define hyperparameters to tune
            params = {}

            params['random_state'] = 1
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['depth'] = trial.suggest_int('depth', 4, 10)
            params['l2_leaf_reg'] = trial.suggest_int('l2_leaf_reg', 0, 10)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
            params['num_boost_round'] = trial.suggest_int('num_boost_round', 250, 2_000)

            if (params['grow_policy'] == 'Lossguide') | (params['grow_policy'] == 'SymmetricTree'):
                 params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, int(train_size/100))

            params['bootstrap_type'] = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No'])

            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)

            if (params['bootstrap_type'] == 'Bernoulli') | (params['bootstrap_type'] == 'MVS'):
                params['subsample'] = trial.suggest_float('subsample', 0.1, 1)

            params['sampling_frequency'] = trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel'])

            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1, sum(ytrain == 0)/sum(ytrain == 1))

            params['verbose'] = False

            early_stopping_rounds = int(params['num_boost_round']*0.3)

            train_pool = catboost.Pool(data = xtrain, label = ytrain, feature_names = list(xtrain.columns))
            del xtrain, ytrain

            val_pool = catboost.Pool(data = xval, label = yval, feature_names = list(xval.columns))
            del xval, yval

            try:
                # Initialize Catboost classifier with current hyperparameters
                clf = catboost.CatBoostClassifier(**params)
                # Train the classifier
                clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds, verbose=False, plot=False)

                #Evaluation metrics
                preds = clf.predict(val_pool)
                ytrue = val_pool.get_label()

                accuracy = accuracy_score(y_true = ytrue, y_pred = preds)
                balanced_accuracy = balanced_accuracy_score(y_true = ytrue, y_pred = preds)
                precision = precision_score(y_true = ytrue, y_pred = preds)
                recall = recall_score(y_true = ytrue, y_pred = preds)
                f1 = f1_score(y_true = ytrue, y_pred = preds)

                objective_fn = (balanced_accuracy + precision + f1)/3

            except:
                 accuracy = 0
                 balanced_accuracy = 0
                 precision = 0
                 recall = 0
                 f1 = 0     
                 objective_fn = 0


            # Log hyperparameters and metrics to W&B
            wandb.log({
                 'trial': trial.number,
                 'balanced_accuracy': balanced_accuracy,
                 'accuracy': accuracy,
                 'precision': precision,
                 'recall': recall,
                 'f1': f1,
                 'objective_fn': objective_fn,
                })

            # Return the metric to optimize (e.g., accuracy)
            return objective_fn

        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial = trial, dataset = self.dataset), n_trials=10)

        self.study = study

        self.next(self.train_final_model)

    @step
    def train_final_model(self):
        """
        This step is used to train the final model with the best hyperparameters on the full dataset. 
        """
        import pandas as pd
        import catboost
        import numpy as np
        best_params = self.study.best_params
        xtrain, ytrain = self.dataset['xtrain'], self.dataset['ytrain']
        xval, yval = self.dataset['xval'], self.dataset['yval']

        #Concatenate train and validation data
        xtrain = pd.concat([xtrain, xval])
        ytrain = np.concatenate([ytrain, yval])

        train_pool = catboost.Pool(data = xtrain, label = ytrain, feature_names = list(xtrain.columns))
        del xtrain, ytrain

        # Initialize Catboost classifier with current hyperparameters
        clf = catboost.CatBoostClassifier(**best_params)
        # Train the classifier
        clf.fit(train_pool, verbose=False, plot=False)

        self.model = clf

        self.next(self.end)

    def calibrate_model(self):


        self.next(self.end)


    @step
    def end(self):
        """
        Flow completed
        """
        pass

if __name__ == "__main__":
    TuningFlow()