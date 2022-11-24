from pyHalo.preset_models import preset_model_from_name
import numpy as np
from copy import deepcopy
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size
from pyHalo.realization_extensions import RealizationExtensions

def draw(prior, prior_type):
    if prior_type == 'UNIFORM':
        value = np.random.uniform(prior[1], prior[2])
    elif prior_type == 'GAUSSIAN':
        value = np.random.normal(prior[1], prior[2])
    elif prior_type == 'FIXED':
        value = prior[1]
    elif prior_type == 'DISCRETE':
        value = np.random.choice(prior[1])
    else:
        raise Exception('prior type ' + str(prior_type) + ' not recognized.')
    return value

def setup_realization(priors, x_image=None, y_image=None, source_size_pc=None):

    realization_priors = deepcopy(priors)
    realization_params = None
    kwargs_realization = {}
    preset_model_name = realization_priors['PRESET_MODEL']

    del realization_priors['PRESET_MODEL']
    param_names_realization = []

    for parameter_name in realization_priors.keys():

        prior_type = realization_priors[parameter_name][0]
        prior = realization_priors[parameter_name]
        value = draw(prior, prior_type)
        if parameter_name == 'log10_sigma_sub':
            kwargs_realization['sigma_sub'] = 10 ** value
        else:
            kwargs_realization[parameter_name] = value

        if prior_type == 'FIXED':
            continue
        else:
            param_names_realization.append(parameter_name)

        if realization_params is None:
            realization_params = value
        else:
            realization_params = np.append(realization_params, value)

    if preset_model_name == 'CDM_SCALED':
        preset_model = CDM_scaled
    elif preset_model_name == 'WDM_x':
        preset_model = CUSTOM_WDM
    elif preset_model_name == 'SIDM_CORE_COLLAPSE':
        preset_model = SIDM_CORE_COLLAPSE
    elif preset_model_name == 'ULDM':
        assert x_image is not None
        assert y_image is not None
        assert source_size_pc is not None
        if kwargs_realization['flucs'] is True:
            aperture_radius = auto_raytracing_grid_size(source_size_pc) * 1.25
            kwargs_realization['flucs_args'] = {'x_images': x_image,
                                                'y_images': y_image,
                                                'aperture': aperture_radius}
        else:
            pass

        preset_model = preset_model_from_name(preset_model_name)

    else:
        try:
            preset_model = preset_model_from_name(preset_model_name)
        except:
            raise Exception('preset model '+str(preset_model_name)+' not recognized.')

    return realization_params, preset_model, kwargs_realization, param_names_realization

def CDM_scaled(zlens, zsource, **kwargs_rendering):

    if 'log10_mfunc_scaling' in kwargs_rendering.keys():
        scale = 10**kwargs_rendering['log10_mfunc_scaling']
        del kwargs_rendering['log10_mfunc_scaling']
    else:
        scale = kwargs_rendering['mfunc_scaling']
        del kwargs_rendering['mfunc_scaling']
    sigma_sub = 0.05 * scale
    LOS_norm = 1.0 * scale
    CDM = preset_model_from_name('CDM')
    return CDM(zlens, zsource, sigma_sub=sigma_sub, LOS_normalization=LOS_norm, **kwargs_rendering)

def SIDM_CORE_COLLAPSE(zlens, zsource, **kwargs_rendering):

    CDM = preset_model_from_name('CDM')
    realization_cdm = CDM(zlens, zsource, **kwargs_rendering)
    lens_cosmo = realization_cdm.lens_cosmo
    ext = RealizationExtensions(realization_cdm)
    mass_range = [[6.0, 7.5], [7.5, 8.5], [8.5, 10.0]]
    p675_sub = kwargs_rendering['f_675_sub']
    p7585_sub = kwargs_rendering['f_7585_sub']
    p910_sub = kwargs_rendering['f_8510_sub']

    def _collape_probability_field(z, prob, zlens, lens_cosmo):
        rescale = lens_cosmo.cosmo.halo_age(zlens)/lens_cosmo.cosmo.halo_age(z)
        return prob * rescale

    p1 = min(1.0, kwargs_rendering['r_1_field'] * p675_sub)
    p2 = min(1.0, kwargs_rendering['r_2_field'] * p7585_sub)
    p3 = min(1.0, kwargs_rendering['r_3_field'] * p910_sub)

    kwargs_field_1 = {'prob': p1, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    kwargs_field_2 = {'prob': p2, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    kwargs_field_3 = {'prob': p3, 'zlens': zlens, 'lens_cosmo': lens_cosmo}
    p675_field = _collape_probability_field
    p759_field = _collape_probability_field
    p910_field = _collape_probability_field
    kwargs_field = [kwargs_field_1, kwargs_field_2, kwargs_field_3]

    probabilities_subhalos = [p675_sub, p7585_sub, p910_sub]
    probabilities_field_halos = [p675_field, p759_field, p910_field]

    indexes = ext.core_collapse_by_mass(mass_range, mass_range,
                              probabilities_subhalos, probabilities_field_halos, kwargs_field=kwargs_field)

    if kwargs_rendering['halo_profile'] == 'SPL_CORE':
        kwargs_core_collapse_profile = {'x_match': kwargs_rendering['x_match'],
                                        'x_core_halo': kwargs_rendering['x_core_halo'],
                                        'log_slope_halo': kwargs_rendering['log_slope_halo']}
    elif kwargs_rendering['halo_profile'] == 'GNFW':
        kwargs_core_collapse_profile = {'x_match': kwargs_rendering['x_match'],
                                        'gamma_inner': kwargs_rendering['gamma_inner'],
                                        'gamma_outer': kwargs_rendering['gamma_outer']}
    else:
        raise Exception('halo profile must be specified')

    realization_sidm = ext.add_core_collapsed_halos(indexes, kwargs_rendering['halo_profile'],
                                                    **kwargs_core_collapse_profile)
    return realization_sidm

def CUSTOM_WDM(zlens, zsource, **kwargs_rendering):

    def a_func_mfunc(x, norm=0.33, slope=-0.072):
        y = norm + slope * (x - 0.5)
        return y

    def b_func_mfunc(x, norm=0.28, slope=0.6):
        y = norm + slope * (x - 0.5)
        return y

    def a_func_mcrel(x, norm=0.75, slope=0.4):
        y = norm * (0.7 / (x)) ** slope
        return y

    def b_func_mcrel(x, norm=0.94, slope=0.7, shift=0.6):
        y = norm * abs(np.log(1.76 + shift) / np.log(x + shift)) ** slope
        return y

    x = kwargs_rendering['x_wdm']
    a_wdm = a_func_mfunc(x)
    b_wdm = b_func_mfunc(x)
    c_wdm = -3.
    a_mc = a_func_mcrel(x)
    b_mc = b_func_mcrel(x)

    kwargs_rendering['a_wdm_los'] = a_wdm
    kwargs_rendering['b_wdm_los'] = b_wdm
    kwargs_rendering['c_wdm_los'] = c_wdm
    kwargs_rendering['a_wdm_sub'] = a_wdm
    kwargs_rendering['b_wdm_sub'] = b_wdm
    kwargs_rendering['c_wdm_sub'] = c_wdm
    kwargs_rendering['kwargs_suppression_mc_relation_field'] = {'a_mc': a_mc, 'b_mc': b_mc}
    kwargs_rendering['kwargs_suppression_mc_relation_sub'] = {'a_mc': a_mc, 'b_mc': b_mc}
    kwargs_rendering['suppression_model_field'] = 'hyperbolic'
    kwargs_rendering['suppression_model_sub'] = 'hyperbolic'

    WDM = preset_model_from_name('WDM')

    return WDM(zlens, zsource, **kwargs_rendering)
