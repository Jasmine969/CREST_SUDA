from brian2modelfitting import MSEMetric, FeatureMetric, calc_eFEL
from brian2 import *
import warnings
import efel
from itertools import repeat


class CaLMetric(MSEMetric):
    def __init__(self, nan_replace, t_wt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_replace = nan_replace
        self.t_wt = t_wt

    def get_features(self, model_traces, data_traces, dt):
        error = (model_traces - data_traces) ** 2
        error *= self.t_wt
        error = error.mean(axis=2)
        v_peak = model_traces[:, 1, 16000:22000].max(axis=-1)
        v_peak_max = 0 / 1000
        v_peak_min = -20 / 1000
        err_peak = ((abs(v_peak - v_peak_max) + abs(v_peak - v_peak_min)
                     + v_peak_min - v_peak_max) * 0.5) ** 2
        v_minimal = model_traces[:, 1, 17000:22000].min(axis=-1)
        v_minimal_min = -57 / 1000
        err_minimal = clip(v_minimal_min - v_minimal, 0, 1e9) ** 2
        error[:, 1] += (err_peak + err_minimal) * 0.1
        return error

    def get_errors(self, features):
        errors = (features * array([[0.6, 0.4]])).sum(axis=1)
        errors[isnan(errors)] = self.nan_replace
        return errors


class MyFeatureMetric(FeatureMetric):
    def __init__(self, nan_replace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(nan_replace, int) or isinstance(nan_replace, float):
            self.nan_replace = dict()
            for k in self.feat_list:
                self.nan_replace[k] = nan_replace
        else:
            assert isinstance(nan_replace, dict)
            self.nan_replace = nan_replace
        if not isinstance(self.combine, dict):
            self.comb_dct = dict()
            for k in self.feat_list:
                self.comb_dct[k] = self.combine
            self.combine = self.comb_dct

    def check_values(self, feat_list):
        """Removes all the None values and checks for array features"""
        for r in feat_list:
            for k, v in r.items():
                if v is None or (len(r[k]) == 1 and isnan(r[k])):
                    r[k] = array([self.nan_replace[k]])
                    warnings.warn('None for key:{}'.format(k))
                if len(r[k]) > 1:
                    r[k] = r[k].mean(keepdims=True)

    def feat_to_err(self, d1, d2):
        d = {}
        err = 0
        for key in d1.keys():
            x = d1[key]
            y = d2[key]
            d[key] = self.combine[key](x, y)
        return d


def err_rel(y_pred, y_exp):
    return abs((y_pred - y_exp) / y_exp)


def err_rel_positive(y_pred, y_exp):
    return clip((y_pred - y_exp) / y_exp, 0, inf)


def err_rel_custom(y_pred, y_exp):
    return abs((y_pred - y_exp) / 10)


def err_range(y_pred, ind, ymin, ymax, y_base, nan_replace, name):
    y_min = ymin[ind]
    y_max = ymax[ind]
    assert y_min <= y_max
    if y_pred is None or (len(y_pred) == 1 and isnan(y_pred)):
        y_pred = array([nan_replace])
        warnings.warn(f'None for key:{name}')
    if y_pred < y_min:
        return (y_min - y_pred) / y_base
    if y_pred > y_max:
        return (y_pred - y_max) / y_base
    return array([0])
