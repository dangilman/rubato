from rubato.inference.util import *
from rubato.workflow import single_iteration_from_samples
from rubato.inference.setup_realization import setup_realization, draw
from scipy.stats.kde import gaussian_kde
import os
import subprocess
from lenstronomy.LensModel.lens_model import LensModel

def forward_model(output_path, mcmc_samples_array, param_class, job_index, flux_ratios_data, n_keep,
                  kwargs_sample_realization, kwargs_sample_redshifts,
                  kwargs_sample_source, kwargs_sample_matching_scale, lens_model_list_macro_other=[],
                  redshift_list_macro_other=[], kwargs_lens_macro_other=[],
                  tolerance=0.5, verbose=False, readout_steps=2,
                  lens_model_list_estimate=False, index_model_list=None,
                  uncertainty_in_magnifications=True, flux_uncertanties=None, keep_flux_ratio_index=[0,1,2],
                  test_mode=False):

    # set up the filenames and folders for writing output
    filename_parameters, filename_mags, filename_realizations, filename_sampling_rate, filename_acceptance_ratio = \
        filenames(output_path, job_index)

    # if the required directories do not exist, create them
    if os.path.exists(output_path) is False:
        proc = subprocess.Popen(['mkdir', output_path])
        proc.wait()
    if os.path.exists(output_path + 'job_' + str(job_index)) is False:
        proc = subprocess.Popen(['mkdir', output_path + 'job_' + str(job_index)])
        proc.wait()

    if verbose:
        print('reading output to files: ')
        print(filename_parameters)
        print(filename_mags)

    # You can restart inferences from previous runs by simply running the function again. In the following lines, the
    # code looks for existing output files, and determines how many samples to add based on how much output already
    # exists.
    if os.path.exists(filename_mags):
        _m = np.loadtxt(filename_mags)
        try:
            n_kept = _m.shape[0]
        except:
            n_kept = 1
        write_param_names = False
    else:
        n_kept = 0
        _m = None
        write_param_names = True

    if n_kept >= n_keep:
        print('\nSIMULATION ALREADY FINISHED.')
        return

    # Initialize stuff for the inference
    idx_init = deepcopy(n_kept)
    parameter_array = None
    mags_out = None
    readout = False
    break_loop = False
    saved_lens_systems = []
    accepted_realizations_counter = 0

    if verbose:
        print('starting with ' + str(n_kept) + ' samples accepted, ' + str(n_keep - n_kept) + ' remain')
        print('existing magnifications: ', _m)
        print('samples remaining: ', n_keep - n_kept)

    #_flux_ratios_data = magnifications[1:] / magnifications[0]
    mcmc_samples_kde = gaussian_kde(mcmc_samples_array.T)
    # start the simulation, the while loop will execute until one has obtained n_keep samples from the posterior
    while True:

        mcmc_sample = np.squeeze(mcmc_samples_kde.resample(size=1))
        log10_angular_matching_scale = draw(kwargs_sample_matching_scale['log10_matching_scale'],
                                      kwargs_sample_matching_scale['log10_matching_scale'][0])
        angular_matching_scale = 10 ** log10_angular_matching_scale

        zlens = draw(kwargs_sample_redshifts['zlens'], kwargs_sample_redshifts['zlens'][0])
        zsource = draw(kwargs_sample_redshifts['zsource'], kwargs_sample_redshifts['zsource'][0])

        # Now, setup the source model, and ray trace to compute the image magnifications
        source_size_pc = draw(kwargs_sample_source['source_size_pc'], kwargs_sample_source['source_size_pc'][0])

        # parse the input dictionaries into arrays with parameters drawn from their respective priors
        realization_samples, preset_model, kwargs_preset_model, param_names_realization = setup_realization(
            kwargs_sample_realization)

        realization = preset_model(zlens, zsource, **kwargs_preset_model)
        cosmo = realization.lens_cosmo.cosmo.astropy
        lens_model_estimate = LensModel(lens_model_list_estimate,
                                        lens_redshift_list=[zlens]*2,
                                        z_source=zsource, multi_plane=True,
                                        cosmo=cosmo)
        keep_images = [True]*4
        tan_diff = False
        filter_low_mass = True
        kwargs_filter = {}
        align_with_source = True
        finite_area = False
        return_flux_ratios = False
        return_full_output = True

        if verbose:
            print('realization contains ' + str(len(realization.halos)) + ' halos.')
            print(param_names_realization)
            print(realization_samples)
            print('source size: ', source_size_pc)
            print('log10 angular matching scale: ', log10_angular_matching_scale)

        mags, multi_image_model, kwargs_arc = single_iteration_from_samples(angular_matching_scale, source_size_pc,
                                                    lens_model_estimate, param_class, mcmc_sample,
                                                    zlens, zsource, cosmo, keep_images,
                                                    realization, lens_model_list_macro_other,
                                                    redshift_list_macro_other, kwargs_lens_macro_other,
                                                    tan_diff, filter_low_mass, kwargs_filter, align_with_source,
                                                    finite_area, index_model_list, return_flux_ratios, return_full_output,
                                                                            verbose)
        mags *= mags[0]**-1
        # if test_mode:
        #     multi_image_model.plot_images(source_size_pc, kwargs_arc, grid_size=0.125, grid_resolution=0.002)
        #     multi_image_model.plot_kappa(kwargs_arc, grid_size=0.125, grid_resolution=0.005)
        #     _=input('proceed?')

        if uncertainty_in_magnifications:
            mags_with_uncertainties = []
            for j, mag in enumerate(mags):
                if flux_uncertanties[j] is None:
                    m = np.nan
                else:
                    m = abs(mag + np.random.normal(0, flux_uncertanties[j] * mag))
                mags_with_uncertainties.append(m)
            mags_with_uncertainties = np.array(mags_with_uncertainties)
            _flux_ratios = mags_with_uncertainties[1:] / mags_with_uncertainties[0]
        else:
            # If uncertainties are quoted for image flux ratios, we first compute the flux ratios, and then add
            # the uncertainties
            flux_ratios = mags[1:] / mags[0]
            fluxratios_with_uncertainties = []

            for k, fr in enumerate(flux_ratios):
                if flux_uncertanties[k] is None:
                    new_fr = np.nan
                else:
                    df = np.random.normal(0, fr * flux_uncertanties[k])
                    new_fr = fr + df

                fluxratios_with_uncertainties.append(new_fr)
            _flux_ratios = np.array(fluxratios_with_uncertainties)

        fr_data = []
        flux_ratios = []
        for idx in keep_flux_ratio_index:
            flux_ratios.append(_flux_ratios[idx])
            fr_data.append(flux_ratios_data[idx])

        # Now we compute the summary statistic
        stat = 0
        for f_i_data, f_i_model in zip(fr_data, flux_ratios):
            stat += (f_i_data - f_i_model) ** 2
        stat = np.sqrt(stat)

        if verbose:
            print('flux ratios data: ', flux_ratios_data)
            print('flux ratios model: ', _flux_ratios)
            print('statistic: ', stat)

        if stat < tolerance:
            # If the statistic is less than the tolerance threshold, we keep the parameters
            accepted_realizations_counter += 1
            n_kept += 1
            params = np.append(np.append(np.append(realization_samples, source_size_pc), log10_angular_matching_scale), stat)
            param_names = param_names_realization + ['source_size_pc'] + ['log10_angular_match_scale'] + [
                'summary_statistic']

            if parameter_array is None:
                parameter_array = params
            else:
                parameter_array = np.vstack((parameter_array, params))

            if mags_out is None:
                mags_out = mags
            else:
                mags_out = np.vstack((mags_out, mags))

            if verbose:
                print('N_kept: ', n_kept)
                print('N remaining: ', n_keep - n_kept)

        if accepted_realizations_counter == readout_steps:
            readout = True
            accepted_realizations_counter = 0
        # break loop if we have collected n_keep samples
        if n_kept == n_keep:
            readout = True
            break_loop = True

        if readout:
            # Now write stuff to file
            readout = False

            with open(filename_parameters, 'a') as f:
                if write_param_names:
                    param_name_string = ''
                    for name in param_names:
                        param_name_string += name + ' '
                    f.write(param_name_string + '\n')
                    write_param_names = False

                nrows, ncols = int(parameter_array.shape[0]), int(parameter_array.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(parameter_array[row, col], 6)) + ' ')
                    f.write('\n')

            with open(filename_mags, 'a') as f:
                nrows, ncols = int(mags_out.shape[0]), int(mags_out.shape[1])
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        f.write(str(np.round(mags_out[row, col], 6)) + ' ')
                    f.write('\n')

            idx_init += len(saved_lens_systems)
            parameter_array = None
            mags_out = None
            saved_lens_systems = []

        if break_loop:
            print('\nSIMULATION FINISHED')
            break
