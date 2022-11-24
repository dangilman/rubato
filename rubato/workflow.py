import numpy as np
from rubato.rubato import MultiImageModel

def parallel_iteration(ncpu, args_list):

    from multiprocess.pool import Pool

    pool = Pool(ncpu)
    out = pool.map(single_iteration_from_samples_from_args, args_list)
    pool.close()
    flux_ratios = np.empty((len(out),3))

    for i in range(0, len(out)):
        flux_ratios[i,:] = out[i]
    return flux_ratios

def single_iteration_from_samples_from_args(args):
    return single_iteration_from_samples(*args)

def single_iteration_from_samples(angular_matching_scale,
                                  source_fwhm_pc,
                                  lens_model_samples,
                                  param_class,
                                  mcmc_sample,
                                  zlens, zsource, cosmo,
                                  keep_images=None,
                                  realization=None, lens_model_list_macro_other=[],
                                  redshift_list_macro_other=[], kwargs_lens_macro_other=[],
                                  tan_diff=False, filter_low_mass=True, kwargs_filter={},
                                  align_with_source=True, finite_area=False, index_model_list=None,
                                  return_flux_ratios=True, return_full_output=False, verbose=False):

    kwargs = param_class.args2kwargs(mcmc_sample)
    kwargs_lens = kwargs['kwargs_lens']
    kwargs_ps = kwargs['kwargs_ps'][0]
    x_images, y_images = kwargs_ps['ra_image'], kwargs_ps['dec_image']
    source_x, source_y = lens_model_samples.ray_shooting(x_images[0], y_images[0], kwargs_lens)

    if realization is None:
        model = MultiImageModel(x_images, y_images, source_x, source_y, lens_model_list_macro_other,
                                redshift_list_macro_other, kwargs_lens_macro_other, zlens, zsource, cosmo, tan_diff,
                                keep_images, finite_area)
    else:
        model = MultiImageModel.from_realization(x_images, y_images, source_x, source_y, realization, zlens, zsource,
                                                 cosmo, tan_diff, keep_images, filter_low_mass, kwargs_filter, align_with_source,
                                                 lens_model_list_macro_other, redshift_list_macro_other, kwargs_lens_macro_other,
                                                 finite_area)

    kwargs_arc, _ = model.solve_kwargs_arc_from_kwargs(lens_model_samples,
                                                       kwargs_lens,
                                                       angular_matching_scale,
                                                       index_model_list, verbose)
    mags = model.mags(kwargs_arc, source_fwhm_pc)
    if return_flux_ratios:
        out = mags[1:]/mags[0]
    else:
        out = mags
    if return_full_output:
        return out, model, kwargs_arc
    else:
        return out
