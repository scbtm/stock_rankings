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

        #dummy train set:
        xtrain_dummy = pd.concat([xtrain, xval, xcal])
        ytrain_dummy = np.concatenate([ytrain, yval, ycal])


        #train a dummy classifier to establish some basic baselines
        #Random classifier
        dummy = DummyClassifier(strategy='uniform')
        dummy.fit(xtrain_dummy, ytrain_dummy)
        dummy_preds = dummy.predict(xtest)

        rd_balanced_accuracy = balanced_accuracy_score(ytest, dummy_preds)
        rd_precision = precision_score(ytest, dummy_preds)
        rd_f1 = f1_score(ytest, dummy_preds)

        metrics = {}
        metrics['random_dummy'] = {'accuracy': accuracy_score(ytest, dummy_preds),
                                   'precision': precision_score(ytest, dummy_preds),
                                   'recall': recall_score(ytest, dummy_preds),
                                   'f1': f1_score(ytest, dummy_preds),
                                   'balanced_accuracy': balanced_accuracy_score(ytest, dummy_preds),
                                   'objective_fn': (rd_balanced_accuracy + rd_precision + rd_f1)/3}
        

        #Most frequent classifier        
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(xtrain_dummy, ytrain_dummy)
        dummy_preds = dummy.predict(xtest)

        del xtrain_dummy, ytrain_dummy

        mfd_balanced_accuracy = balanced_accuracy_score(ytest, dummy_preds)
        mfd_precision = precision_score(ytest, dummy_preds)
        mfd_f1 = f1_score(ytest, dummy_preds) 

        metrics['most_frequent_dummy'] = {'accuracy': accuracy_score(ytest, dummy_preds),
                                          'precision': precision_score(ytest, dummy_preds),
                                          'recall': recall_score(ytest, dummy_preds),
                                          'f1': f1_score(ytest, dummy_preds),
                                          'balanced_accuracy': balanced_accuracy_score(ytest, dummy_preds),
                                          'objective_fn': (mfd_balanced_accuracy + mfd_precision + mfd_f1)/3}
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
        self.n_trials = os.getenv('N_TRIALS')
        gpu_available = os.getenv('GPU_AVAILABLE')

        if gpu_available == 1:
            gpu_available = True
        else:
            gpu_available = False

        # Initialize W&B
        wandb.init(project=project_name, entity=entity)

        def objective(trial, dataset):
            xtrain, ytrain = dataset['xtrain'], dataset['ytrain']
            xval, yval = dataset['xval'], dataset['yval']

            train_size = len(xtrain)
            #Define hyperparameters to tune
            params = {}

            if gpu_available:
                params['task_type'] = 'GPU'

            params['random_state'] = 1
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['depth'] = trial.suggest_int('depth', 4, 10)
            params['l2_leaf_reg'] = trial.suggest_int('l2_leaf_reg', 0, 10)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
            params['num_boost_round'] = trial.suggest_int('num_boost_round', 250, 2_000)

            if (params['grow_policy'] == 'Lossguide') | (params['grow_policy'] == 'SymmetricTree'):
                 params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, int(train_size*0.001))

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
        study.optimize(lambda trial: objective(trial = trial, dataset = self.dataset), n_trials=self.n_trials)

        self.study = study

        self.next(self.train_best_model)

    @step
    def train_best_model(self):
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

        self.next(self.calibrate_model)

    @step
    def calibrate_model(self):

        from sklearn.isotonic import IsotonicRegression
        import catboost

        model = self.model
        xcal, ycal = self.dataset['xcal'], self.dataset['ycal']

        cal_pool = catboost.Pool(data = xcal, label = ycal, feature_names = list(xcal.columns))

        predicted_proba = model.predict_proba(cal_pool)[:, 1]

        iso_reg = IsotonicRegression(y_min = 0, y_max = 1, out_of_bounds = 'clip').fit(predicted_proba, ycal)

        self.calibration_model = iso_reg

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        This step is used to evaluate the model on the test dataset. 
        """
        def expected_calibration_error(y, proba, bins='fd'):
            import numpy as np  # Import numpy for numerical operations

            # Compute the histogram of predicted probabilities to determine the bins
            bin_count, bin_edges = np.histogram(proba, bins=bins)
            n_bins = len(bin_count)  # Number of bins

            # Adjust the first bin edge slightly to include the exact minimum probability
            bin_edges[0] -= 1e-8

            # Assign each probability to a bin
            bin_id = np.digitize(proba, bin_edges, right=True) - 1

            # Calculate the sum of true labels for each bin
            bin_ysum = np.bincount(bin_id, weights=y, minlength=n_bins)

            # Calculate the sum of probabilities for each bin
            bin_probasum = np.bincount(bin_id, weights=proba, minlength=n_bins)

            # Calculate the average of true labels in each bin
            bin_ymean = np.divide(bin_ysum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)

            # Calculate the average of probabilities in each bin
            bin_probamean = np.divide(bin_probasum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)

            # Calculate the Expected Calibration Error (ECE)
            ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)

            # Return the ECE
            return ece
        

        import pandas as pd
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

        import logging

        model = self.model
        calibration_model = self.calibration_model
        xtest, ytest = self.dataset['xtest'], self.dataset['ytest']
        

        predicted_proba = model.predict_proba(xtest)[:, 1]
        predicted_proba = calibration_model.predict(predicted_proba)
        preds = (predicted_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_true = ytest, y_pred = preds)
        balanced_accuracy = balanced_accuracy_score(y_true = ytest, y_pred = preds)
        precision = precision_score(y_true = ytest, y_pred = preds)
        recall = recall_score(y_true = ytest, y_pred = preds)
        f1 = f1_score(y_true = ytest, y_pred = preds)

        ece = expected_calibration_error(ytest, predicted_proba)

        objective_fn = (balanced_accuracy + precision + f1)/3

        self.metrics = {'accuracy': accuracy,
                        'balanced_accuracy': balanced_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'ece': ece,
                        'objective_fn': objective_fn}
        

        #Ece of final model vs random predictions
        random_preds = np.random.uniform(0, 1, len(ytest))
        random_ece = expected_calibration_error(ytest, random_preds)

        logging.info(f'Final model ECE: {ece}')
        logging.info(f'Random model ECE: {random_ece}')
        

        #Check how much lift the final model has on baseline dummy models for each of the baseline metrics
        final_metrics = self.metrics
        baseline_metrics = self.baseline_metrics

        for dummy_model in baseline_metrics.keys():
            for metric in baseline_metrics[dummy_model].keys():

                lift = final_metrics[metric] - baseline_metrics[dummy_model][metric]

                logging.info(f'Final model performance on {metric}: {final_metrics[metric]}')
                logging.info(f'Final model lift over {dummy_model}_{metric}: {lift}')


        self.next(self.package_models)

    

    @step
    def package_models(self):
        """
        This step is used to save the models to disk. 
        """
        import pandas as pd
        import joblib
        from dotenv import load_dotenv
        import os
        import wandb
        # Load environment variables from .env
        load_dotenv()

        #Path to artifact registry
        artifact_registry = os.getenv('ARTIFACT_REGISTRY')
        project = os.getenv('WANDB_PROJECT_NAME')
        entity = os.getenv('WANDB_ENTITY')

        # Save the models
        joblib.dump(self.model, f'{artifact_registry}/main_model.pkl')
        joblib.dump(self.calibration_model, f'{artifact_registry}/calibration_model.pkl')
        #save optuna study
        study = self.study
        joblib.dump(study, f"{artifact_registry}/main_model_optuna_study.pkl")

        

        # Initialize a W&B run
        run = wandb.init(project=project, entity=entity, job_type='save-model')
        #run.link_model(name="StockRanker", path=artifact_registry, registered_model_name="StockRanker")

        
        # Create an artifact for the model
        artifact = wandb.Artifact("StockRanker", type="model")

        # Add a file or directory to the artifact
        artifact.add_dir(artifact_registry)  # Assuming artifact_registry is a directory with your model files

        # Log the artifact
        run.log_artifact(artifact)
        

        # Optional: log additional metrics or information
        run.log(self.metrics)

        today = pd.Timestamp.today().strftime('%Y-%m-%d')

        run.notes = f'Results of hyperparameter tuning stage. Date: {today}'

        # Finish the run
        run.finish()


        

        """
        To log and link the model to the wandb model registry, use the following code (testing may be required):
        import wandb

        # Initialize a W&B run
        run = wandb.init(project=project, entity=entity, job_type='save-model')

        # Log the model
        #run.log_model(path=artifact_registry, name="StockRanker")
        run.log(self.metrics)
        run.link_model(name="StockRanker", path=artifact_registry, registered_model_name="StockRanker")

        run.finish()
        """

        self.next(self.end)

    @step
    def end(self):
        """
        Flow completed
        """
        pass


if __name__ == "__main__":
    TuningFlow()