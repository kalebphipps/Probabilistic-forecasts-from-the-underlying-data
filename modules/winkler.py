import logging
from typing import Dict, Callable, Optional

import xarray as xr
import numpy as np
from pywatts.core.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList, SummaryObject

import pandas as pd


logger = logging.getLogger(__name__)


class winkler_score(BaseSummary):

    def __init__(self, name: str = "Winkler_Score",
                 quantiles: list = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles

    def calculate_quantile_score(self,quant,forecast,obs):
        err = forecast-obs
        ql = np.maximum(quant*err, (quant-1)*err)
        return 2*ql

    def get_params(self) -> Dict[str, object]:

        return {"quantiles": self.quantiles}

    def transform(self, file_manager, y: xr.DataArray, **kwargs:xr.DataArray) -> SummaryObject:

        summary = SummaryObjectList(self.name)

        t = y.values.reshape(y.shape[:-1])
        for key, y_hat in kwargs.items():
            q1 = self.calculate_quantile_score(quant=0.01, forecast=y_hat.sel(quantiles=1), obs=t)
            q5 = self.calculate_quantile_score(quant=0.05, forecast=y_hat.sel(quantiles=5), obs=t)
            q10 = self.calculate_quantile_score(quant=0.10, forecast=y_hat.sel(quantiles=10), obs=t)
            q15 = self.calculate_quantile_score(quant=0.15, forecast=y_hat.sel(quantiles=15), obs=t)
            q20 = self.calculate_quantile_score(quant=0.2, forecast=y_hat.sel(quantiles=20), obs=t)
            q25 = self.calculate_quantile_score(quant=0.25, forecast=y_hat.sel(quantiles=25), obs=t)
            q30 = self.calculate_quantile_score(quant=0.30, forecast=y_hat.sel(quantiles=30), obs=t)
            q40 = self.calculate_quantile_score(quant=0.40, forecast=y_hat.sel(quantiles=40), obs=t)
            q60 = self.calculate_quantile_score(quant=0.60, forecast=y_hat.sel(quantiles=60), obs=t)
            q70 = self.calculate_quantile_score(quant=0.70, forecast=y_hat.sel(quantiles=70), obs=t)
            q75 = self.calculate_quantile_score(quant=0.75, forecast=y_hat.sel(quantiles=75), obs=t)
            q80 = self.calculate_quantile_score(quant=0.80, forecast=y_hat.sel(quantiles=80), obs=t)
            q85 = self.calculate_quantile_score(quant=0.85, forecast=y_hat.sel(quantiles=85), obs=t)
            q90 = self.calculate_quantile_score(quant=0.90, forecast=y_hat.sel(quantiles=90), obs=t)
            q95 = self.calculate_quantile_score(quant=0.95, forecast=y_hat.sel(quantiles=95), obs=t)
            q99 = self.calculate_quantile_score(quant=0.99, forecast=y_hat.sel(quantiles=99), obs=t)

            w2 = (q1+q99)/0.02
            w10 = (q5+q95)/0.1
            w20 = (q10+q90)/0.2
            w30 = (q15+q85)/0.3
            w40 = (q20+q80)/0.4
            w50 = (q25 + q75)/0.5
            w60 = (q30 + q70)/0.6
            w80 = (q40 + q60)/0.8
            winkler_scores = dict({"Winkler2": np.mean(w2.values),
                                   "Winkler10": np.mean(w10.values),
                                   "Winkler20": np.mean(w20.values),
                                   "Winkler30": np.mean(w30.values),
                                   "Winkler40": np.mean(w40.values),
                                   "Winkler50": np.mean(w50.values),
                                   "Winkler60": np.mean(w60.values),
                                   "Winkler80": np.mean(w80.values)})
            summary.set_kv(key, winkler_scores)


        return summary

    def set_params(self, quantiles: Optional[list] = None):

        if quantiles:
            self.quantiles = quantiles