from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, _get_time_indexes


class PredictionIntervalEmpirical(BaseEstimator):
    """
     Module which calculates the prediction intervals for a given point prediction using various conformal prediction ideas from the magpie library

    """

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def __init__(self, name: str = "PredictionIntervalEmpirical", quantiles =[50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles


    def transform(self, prediction: xr.DataArray) -> xr.DataArray:
        """

        """
        prediction_quantiles = {}

        for q in self.quantiles:
            if q == 50:
                prediction_quantiles[q] = prediction.values
            if q > 50:
                norm_q = q/100
                alpha = 2*(1-norm_q)
                conform_quantile = np.quantile(self.conformity_scores, 1-alpha)

                prediction_quantiles[q] = prediction.values + conform_quantile
                lower_q = 100 - q
                prediction_quantiles[lower_q] = prediction.values - conform_quantile

        arr = np.array(list(prediction_quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(prediction)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(prediction_quantiles.keys()),
                                  _get_time_indexes(prediction)[0]: prediction.indexes[
                                      _get_time_indexes(prediction)[0]]})
        return da

    def fit(self, prediction, target):
        prediction = prediction.values.reshape((len(prediction), -1))
        target = target.values.reshape((len(target), -1))
        self.conformity_scores = np.abs(np.subtract(prediction, target))
        self.is_fitted = True

