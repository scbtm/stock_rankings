from metaflow import FlowSpec, step


class BatchInferenceFlow(FlowSpec):
    @step
    def start(self):
        """
        This step is used to get the data prepared for development. This step is the starting point of the flow. 
        """
        from data_pipeline.pipeline import BatchProductionPipeline

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
        dataset = BatchProductionPipeline(input_data_path=input_data_path, data_source=data_source).run()

        #Get the data
        xinf, yinf = dataset.data['full_df'].drop(['target', 'Date', 'Ticker']), (dataset.data['full_df']['target'] >= target_threshold).cast(pl.Int64)

        xinf = xinf.to_pandas()
        yinf = yinf.to_pandas().values

        dataset = {}
        dataset['xinf'] = xinf
        dataset['yinf'] = yinf

        self.dataset = dataset


        self.next(self.load_latest_model)

    @step
    def load_latest_model(self):

        from dotenv import load_dotenv
        import os
        import joblib
        import wandb

        # Load environment variables from .env
        load_dotenv()

        project = os.getenv('WANDB_PROJECT_NAME')
        entity = os.getenv('WANDB_ENTITY')

        import wandb
        run = wandb.init(project=project, entity=entity, job_type='stage-model')
        artifact = run.use_artifact('san-cbtm/stock-ranking/StockRanker:latest', type='model')
        artifact_dir = artifact.download()

        #Load the model
        self.model = joblib.load(f'{artifact_dir}/main_model.pkl')
        self.calibration_model = joblib.load(f'{artifact_dir}/calibration_model.pkl')


        self.next(self.make_inference)

    @step
    def make_inference(self):
        """
        This step is used to make inference on the dataset. 
        """
        import numpy as np

        xinf = self.dataset['xinf']

        predicted_proba = self.model.predict_proba(xinf)[:, 1]
        predicted_proba = self.calibration_model.predict(predicted_proba)
        preds = (predicted_proba >= 0.5).astype(int)

        self.predictions = preds
        self.predicted_probas = predicted_proba

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
        import polars as pl
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

        ytest = self.dataset['yinf']
        predicted_proba = self.predicted_probas
        preds = self.predictions
        
        #The most recent observations in ytest are null values, so we need to remove them and compare predictions to the ones we do have
        
        predictions_df = pl.DataFrame({'ytest': ytest, 'predicted_proba': predicted_proba, 'preds': preds})

        self.predictions_df = predictions_df

        monitoring_df = predictions_df.filter(pl.col('ytest').is_not_null())

        monitoring_df = monitoring_df.filter(pl.col('ytest').is_not_nan())

        ytest = monitoring_df['ytest'].to_numpy()
        predicted_proba = monitoring_df['predicted_proba'].to_numpy()
        preds = monitoring_df['preds'].to_numpy()


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

        self.next(self.save_outputs)

    @step
    def save_outputs(self):
        """
        This step is used to save the predictions and monitoring metrics. 
        """
        import pandas as pd
        from dotenv import load_dotenv
        import os
        import wandb
        # Load environment variables from .env
        load_dotenv()

        project = os.getenv('WANDB_PROJECT_NAME')
        entity = os.getenv('WANDB_ENTITY')
        output_data_path = os.getenv('OUTPUT_DATA_PATH')
        

        # Initialize a W&B run
        run = wandb.init(project=project, entity=entity, job_type='log-metrics')

        run.log(self.metrics)

        today = pd.Timestamp.today().strftime('%Y-%m-%d')

        self.today = today

        run.notes = f"Batch inference run. Date of inference: {today}"

        # Finish the run
        run.finish()

        #save data
        predictions_df = self.predictions_df.to_pandas()
        predictions_df.to_csv(output_data_path, index = False)



        self.next(self.end)

    @step
    def end(self):
        """
        Flow completed
        """
        print(f"Flow completed. Predictions made on batch for date {self.today}.")


if __name__ == "__main__":
    BatchInferenceFlow()