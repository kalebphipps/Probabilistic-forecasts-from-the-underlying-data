from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, _get_time_indexes
from scipy.stats import norm


class PredictionIntervalGaussian(BaseEstimator):
    """
     Module which calculates the prediction intervals for a given point prediction under the assumption of a normally distributed forecast error

    :param name: The name of the module
    :type name: str

     :param prediction: Name of the new variable
     :type name: str

    """

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def __init__(self, name: str = "PredictionIntervalGaussian", quantiles =[50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles


    def transform(self, prediction: xr.DataArray) -> xr.DataArray:
        """
        Calculates the prediction intervals for a given point prediction under the assumption of a normally distributed forecast error.

        :param y_df: the target values
        :type y_df: xr.DataArray

        :param prediction: the predcition values
        :type prediction: xr.DataArray

        :return: The calculated prediction intervals
        :rtype: xr.DataArray
        """
        prediction_quantiles = { }

        for q in self.quantiles:
            if q == 50:
                prediction_quantiles[q] = prediction.values
            if q > 50:
                multiplier = q / 100
                multiplier = 1 - multiplier
                multiplier = norm.ppf(1 - (multiplier / 2))

                prediction_quantiles[q] = prediction.values + self.rstd * multiplier
                lower_q = 100-q
                prediction_quantiles[lower_q] = prediction.values - self.rstd * multiplier


        arr = np.array(list(prediction_quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(prediction)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(prediction_quantiles.keys()),
                                  _get_time_indexes(prediction)[0]: prediction.indexes[_get_time_indexes(prediction)[0]]})
        return da

    def fit(self, prediction, target):
        prediction = prediction.values.reshape((len(prediction), -1))
        target = target.values.reshape((len(target), -1))
        self.rstd = np.sqrt((np.mean(np.square(prediction - target))))
        self.is_fitted = True

