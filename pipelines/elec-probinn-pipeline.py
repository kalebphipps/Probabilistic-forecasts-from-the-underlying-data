import os

import pandas as pd

from pywatts.modules import CalendarExtraction, CalendarFeature, SKLearnWrapper, KerasWrapper
from pywatts.core.summary_formatter import SummaryJSON

from modules.inn import INNWrapper
from modules.pytorch_forecasting_determ_wrapper import PyTorchForecastingDeterministicWrapper

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_forecasting.models import NHiTS, TemporalFusionTransformer

from base_pipelines.generic_probINN import train_forecast_pipeline, train_cinn_pipeline, get_prob_forecast_pipeline, \
    get_pi_benchmark_pipeline, flatten, get_keras_model

PIPELINE_NAME = "elec_probinn"
TARGET = "MT_158"
FEATURES = []

QUANTILES = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
INN_STDS = [0.55, 0.80, 0.53, 0.39, 0.64, 0.74]
INN_EPOCHS = 100
INN_SAMPLE_SIZE = 100
NUMER_OF_RUNS = 5


def prepare_data():
    data = pd.read_csv("../data/elec.csv", index_col="time", parse_dates=True)
    data.index.name = 'time'

    train = data.iloc[:14717, :]
    val = data.iloc[14717:21024, :]
    test = data.iloc[21024:26280, :]

    return data, train, val, test


if __name__ == "__main__":
    # Split Data
    data, train, val, test = prepare_data()

    df_pi_bench = pd.DataFrame()
    df_eval = pd.DataFrame()

    for i in range(NUMER_OF_RUNS):

        neural_network = get_keras_model(FEATURES)

        # Build Estimators
        sklearn_estimators = [SKLearnWrapper(module=LinearRegression(), name="Linear_regression"),
                              SKLearnWrapper(module=RandomForestRegressor(), name="RF_Regression"),
                              SKLearnWrapper(module=MLPRegressor(), name="MLP_Regression"),
                              KerasWrapper(neural_network, fit_kwargs={"batch_size": 100, "epochs": 100},
                                           compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]},
                                           name="NN_Regression"),
                              SKLearnWrapper(module=XGBRegressor(), name="XGB_Regression")]

        pytorch_estimators = [PyTorchForecastingDeterministicWrapper(NHiTS, name="N-HITS"),
                              PyTorchForecastingDeterministicWrapper(TemporalFusionTransformer, name="TFT")]

        # Calendar Information
        calendar_extraction = CalendarExtraction(continent="Europe",
                                                 country="Germany",
                                                 features=[CalendarFeature.workday, CalendarFeature.hour_cos,
                                                           CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                                           CalendarFeature.month_cos])

        # Build deterministic forecasting pipeline and train
        deterministic_forecasting_pipeline, target_scaler, feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler = train_forecast_pipeline(
            pipeline_name=PIPELINE_NAME,
            target=TARGET,
            sklearn_estimators=sklearn_estimators,
            pytorch_estimators=pytorch_estimators,
            calendar_extraction=calendar_extraction,
            features=FEATURES)
        deterministic_forecasting_pipeline.train(data=train, summary=True, summary_formatter=SummaryJSON())

        # Create PI_Benchmark pipeline
        pi_benchmark_pipeline, pi_benchmark = get_pi_benchmark_pipeline(pipeline_name=PIPELINE_NAME,
                                                                        target=TARGET,
                                                                        calendar_extraction=calendar_extraction,
                                                                        target_scaler=target_scaler,
                                                                        sklearn_estimators=sklearn_estimators,
                                                                        pytorch_estimators=pytorch_estimators,
                                                                        features=FEATURES,
                                                                        feature1_scaler=feature1_scaler,
                                                                        feature2_scaler=feature2_scaler,
                                                                        feature3_scaler=feature3_scaler,
                                                                        feature4_scaler=feature4_scaler,
                                                                        pi_quantiles=QUANTILES)
        pi_benchmark_pipeline.train(data=val, summary=True, summary_formatter=SummaryJSON())
        result_pi_bench, summary_pi_bench = pi_benchmark_pipeline.test(data=test, summary=True,
                                                                       summary_formatter=SummaryJSON())
        df_pi_bench = df_pi_bench.append(flatten(summary_pi_bench["Summary"]), ignore_index=True)

        # Create cINN
        cinn_model = INNWrapper(name="INN", quantiles=QUANTILES, sample_size=INN_SAMPLE_SIZE)

        # Build cINN Pipeline and train
        cinn_pipeline, cinn = train_cinn_pipeline(pipeline_name=PIPELINE_NAME,
                                                  target=TARGET,
                                                  calendar_extraction=calendar_extraction,
                                                  cinn=cinn_model,
                                                  cinn_epochs=INN_EPOCHS,
                                                  features=FEATURES)

        cinn_pipeline.train(data=train, summary=True, summary_formatter=SummaryJSON())

        for sampling_std in INN_STDS:
            prob_forecast_pipeline, prob_forecast = get_prob_forecast_pipeline(pipeline_name=PIPELINE_NAME,
                                                                               target=TARGET,
                                                                               calendar_extraction=calendar_extraction,
                                                                               target_scaler=target_scaler,
                                                                               sklearn_estimators=sklearn_estimators,
                                                                               pytorch_estimators=pytorch_estimators,
                                                                               cinn_base=cinn,
                                                                               cinn_quantiles=QUANTILES,
                                                                               cinn_sample_size=INN_SAMPLE_SIZE,
                                                                               cinn_sampling_std=sampling_std,
                                                                               features=FEATURES,
                                                                               feature1_scaler=feature1_scaler,
                                                                               feature2_scaler=feature2_scaler,
                                                                               feature3_scaler=feature3_scaler,
                                                                               feature4_scaler=feature4_scaler)

            prob_forecast_pipeline.train(data=val, summary=True, summary_formatter=SummaryJSON())

            result, summary = prob_forecast_pipeline.test(data=test, summary=True, summary_formatter=SummaryJSON())

            df_eval = df_eval.append(flatten(summary["Summary"]), ignore_index=True)

            print("###################################################################################################"
                  "#####################################################################################"
                  "############################################################################################")
            print(f"############################################################################################ "
                  f"For Run {i}, finished STD={sampling_std} ##################################################"
                  f"##########################################")
            print("###################################################################################################"
                  "#####################################################################################"
                  "############################################################################################")

        print("**************************************************************************************************"
              "****************************************************************************************************"
              "****************************************************************** *")
        print(f"************************************************************************************** "
              f"Finished Run {i} ***********************************************************************"
              f"***************************")
        print("**************************************************************************************************"
              "****************************************************************************************************"
              "****************************************************************** *")

    save_path = "../Summaries"
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path)
    df_pi_bench.to_csv(f"{save_path}/pi_benchmark_evaluation_{PIPELINE_NAME}.csv")
    df_eval.to_csv(f"{save_path}/evaluation_{PIPELINE_NAME}.csv")

    print("Finished all runs!")
