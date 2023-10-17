from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, _get_time_indexes


class PredictionIntervalConformal(BaseEstimator):
    """
     Module which calculates the prediction intervals for a given point prediction using various conformal prediction ideas from the magpie library

    """

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def __init__(self, name: str = "PredictionIntervalConformal", quantiles =[50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60],
                 correct=False):
        super().__init__(name)
        self.quantiles = quantiles
        self.correct = correct


    def transform(self, prediction: xr.DataArray) -> xr.DataArray:
        """

        """
        prediction_quantiles = {}

        for q in self.quantiles:
            if q == 50:
                prediction_quantiles[q] = prediction.values
            if q > 50:
                these_errors = self.critical_calibration_score[str(q)]

                prediction_quantiles[q] = prediction.values + these_errors
                lower_q = 100 - q
                prediction_quantiles[lower_q] = prediction.values - these_errors

        arr = np.array(list(prediction_quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(prediction)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(prediction_quantiles.keys()),
                                  _get_time_indexes(prediction)[0]: prediction.indexes[
                                      _get_time_indexes(prediction)[0]]})
        return da

    def fit(self, prediction, target):
        m = prediction.shape[0]
        H = prediction.shape[1]

        prediction = prediction.values.reshape((len(prediction), -1))
        target = target.values.reshape((len(target), -1))
        uncorrected_errors = np.abs(np.subtract(prediction, target))

        self.critical_calibration_score = dict()

        for q in self.quantiles:
            if q > 50:
                norm_q = q/100
                alpha = 2*(1-norm_q)
                if self.correct:
                    corrected_q = min((m + 1.0) * (1 - alpha / H) / m, 1)
                else:
                    corrected_q = min((m + 1.0) * (1 - alpha) / m, 1)
                critical_error = np.quantile(uncorrected_errors, corrected_q, axis=0)
                self.critical_calibration_score[str(q)] = critical_error
        self.is_fitted = True

