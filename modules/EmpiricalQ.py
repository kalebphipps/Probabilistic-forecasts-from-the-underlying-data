import logging
from typing import Dict, Callable, Optional

import xarray as xr
import numpy as np
from pywatts.core.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList, SummaryObject

logger = logging.getLogger(__name__)


class Empirical_Quantile(BaseSummary):

    def __init__(self, name: str = "Empirical_Quantile_Coverage",
                 quantiles: list = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles


    def calculate_empirical_quantile(self, quant, truth, forecast):
        half = np.less_equal(truth, forecast.sel(quantiles=quant))
        return (np.mean(half.sum(axis=1)/24)).values.mean()

    def get_params(self) -> Dict[str, object]:

        return {"quantiles": self.quantiles}

    def transform(self, file_manager, y: xr.DataArray, **kwargs:xr.DataArray) -> SummaryObject:

        summary = SummaryObjectList(self.name)

        for key, y_hat in kwargs.items():
            p = y_hat
            t = y.values.reshape(y.shape[:-1])
            empirical_quantiles = dict()
            for quant in y_hat.quantiles:
                eq = self.calculate_empirical_quantile(quant=quant.values, truth=t, forecast=p)
                this_key = str(quant.values/100)
                empirical_quantiles[this_key] = eq
            summary.set_kv(key, empirical_quantiles)

        return summary

    def set_params(self, quantiles: Optional[list] = None):

        if quantiles:
            self.quantiles = quantiles