"""
Contains functions for optimizing brian2 models
"""
from brian2modelfitting import MSEMetric, FeatureMetric, calc_eFEL
from brian2 import *
import warnings
import efel
from itertools import repeat


class MSENaNMetric(MSEMetric):
    def __init__(self, nan_replace, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_replace = nan_replace

    def get_errors(self, features):
        errors = features.mean(axis=1)
        errors[isnan(errors)] = self.nan_replace
        return errors


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


class FeatureRangeMetric(FeatureMetric):
    """
    The features should fall inside the specified range
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.combine, dict):
            self.comb_dct = dict()
            for k in self.feat_list:
                self.comb_dct[k] = self.combine
            self.combine = self.comb_dct

    def feat_to_err(self, d1, ind):
        d = {}
        for key in d1.keys():
            x = d1[key]
            d[key] = self.combine[key](x, ind)
        return d

    def check_values(self, feat_list):
        """Removes all the None values and checks for array features"""
        for r in feat_list:
            for k, v in r.items():
                if r[k] is not None and len(r[k]) > 1:
                    r[k] = median(r[k], keepdims=True)

    def get_features(self, traces, output, dt):
        n_samples, n_traces, _ = traces.shape
        if len(self.stim_times) != n_traces:
            if len(self.stim_times) == 1:
                self.stim_times = list(repeat(self.stim_times[0], n_traces))
            else:
                raise ValueError("Specify the stim_times variable of appropiate "
                                 "size (same as number of traces or 1).")

        out_feat = calc_eFEL(output, self.stim_times, self.feat_list, dt)
        self.check_values(out_feat)

        features = []
        for one_sample in traces:
            sample_feat = calc_eFEL(one_sample, self.stim_times,
                                    self.feat_list, dt)
            self.check_values(sample_feat)
            sample_features = []
            for ind, one_trace_feat in enumerate(sample_feat):
                sample_features.append(self.feat_to_err(one_trace_feat,
                                                        ind))
            # Convert the list of dictionaries to a dictionary of lists
            sample_features_dict = {}
            for feature_dict in sample_features:
                for key, value in feature_dict.items():
                    if key not in sample_features_dict:
                        sample_features_dict[key] = []
                    if len(value) != 1:
                        raise TypeError('Feature "{}" returned more than a '
                                        'single value, such features are not '
                                        'supported yet.'.format(key))
                    sample_features_dict[key].append(value[0])

            # Convert lists into array
            for key, l in sample_features_dict.items():
                sample_features_dict[key] = array(l)
            features.append(sample_features_dict)

        return features
