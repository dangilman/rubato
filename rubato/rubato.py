import numpy as np
from scipy.optimize import minimize
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from rubato.param_manager import ShiftLensModelParamManager, CurvedArc, CurvedArcTanDiff
from lenstronomy.LensModel.QuadOptimizer.multi_plane_fast import MultiplaneFast
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, auto_raytracing_grid_resolution
from lenstronomy.LightModel.light_model import LightModel
import matplotlib.pyplot as plt
from rubato.Utilities.ray_tracing_util import interpolate_ray_paths
from scipy.interpolate import interp1d
from rubato.Utilities.lens_model_util import MultiplaneFastDifferential
from time import time


class MultiImageModel(object):

    def __init__(self, x_image_list, y_image_list,
                 source_x, source_y,
                 lens_model_list_other, redshift_list_other,
                 kwargs_lens_other, z_lens, z_source, cosmo, tan_diff=False,
                 keep_images=None, finite_area=False):

        if keep_images is None:
            keep_images = [True]*len(x_image_list)
        else:
            assert len(keep_images) == len(x_image_list)

        self._lens_model_list_other = lens_model_list_other
        self._redshift_list_other = redshift_list_other
        self._kwargs_lens_other = kwargs_lens_other
        self._keep_images = keep_images
        self._x_image_list = x_image_list
        self._y_image_list = y_image_list
        self._source_x = source_x
        self._source_y = source_y
        self._zlens = z_lens
        self._zsource = z_source
        self._cosmo = cosmo
        self._tan_diff = tan_diff
        self._finite_area = finite_area

        model_list = []
        for i, (x_img, y_img) in enumerate(zip(self._x_image_list, self._y_image_list)):
            if self._keep_images[i]:
                model = MultiScaleModel(x_img, y_img, source_x, source_y, self._lens_model_list_other,
                                        self._redshift_list_other,
                                        self._kwargs_lens_other, self._zlens, self._zsource, self._cosmo,
                                        self._tan_diff, self._finite_area)
                model_list.append(model)

        self.model_list = model_list
        self.lens_model = model_list[0].lens_model
        self.lens_model_extension = model_list[0].extension

    def set_minimization_function(self, function):

        for model in self.model_list:
            model.set_minimization_function(function)

    @classmethod
    def from_realization(cls, x_image_list, y_image_list,
                 source_x, source_y, realization, z_lens, z_source, cosmo, tan_diff=False,
                         keep_images=None, filter_low_mass=True, kwargs_filter={}, align_with_source=True,
                         lens_model_list_macro_other=[], redshift_list_macro_other=[], kwargs_lens_macro_other=[],
                         finite_area=False):

        _multi_image_model = MultiImageModel(x_image_list, y_image_list, source_x, source_y, lens_model_list_macro_other,
                                             redshift_list_macro_other, kwargs_lens_macro_other, z_lens, z_source, cosmo,
                                             tan_diff, keep_images)
        models = _multi_image_model.model_list

        if align_with_source:
            realization_aligned = align_realization(x_image_list, y_image_list,
                 source_x, source_y, z_source, realization, models)
        else:
            realization_aligned = realization

        if filter_low_mass:
            realization_filtered = filter_realization(x_image_list, y_image_list, source_x, source_y, z_source,
                                                      realization, models, **kwargs_filter)
            lens_model_list_other_filt, redshift_list_other_filt, kwargs_lens_other_filt, _ = realization_filtered.lensing_quantities()
        else:
            lens_model_list_other_filt, redshift_list_other_filt, kwargs_lens_other_filt, _ = realization_aligned.lensing_quantities()
        redshift_list_other_filt = list(redshift_list_other_filt)

        return MultiImageModel(x_image_list, y_image_list, source_x, source_y,
                               lens_model_list_macro_other+lens_model_list_other_filt,
                               redshift_list_macro_other+redshift_list_other_filt,
                               kwargs_lens_macro_other+kwargs_lens_other_filt, z_lens, z_source, cosmo,
                               tan_diff, keep_images, finite_area)

    def plot_kappa(self, kwargs_solution_list, grid_size=0.2, grid_resolution=0.005, kappa_target=None, vmin_max=0.05):

        npix = int(2 * grid_size / grid_resolution)
        _r = np.linspace(-grid_size, grid_size, npix)
        _xx, _yy = np.meshgrid(_r, _r)
        shape0 = _xx.shape
        kappa_list = []
        for i, (xi, yi) in enumerate(zip(self._x_image_list, self._y_image_list)):
            kap = self.lens_model.kappa(xi + _xx.ravel(), yi + _yy.ravel(), kwargs_solution_list[i])
            kappa_list.append(kap.reshape(shape0))
        fig = plt.figure(1)
        fig.set_size_inches(16, 6)
        N = len(self._x_image_list)
        ax1 = plt.subplot(1, N, 1)
        ax2 = plt.subplot(1, N, 2)
        ax3 = plt.subplot(1, N, 3)
        ax4 = plt.subplot(1, N, 4)

        if kappa_target is None:
            kappa_target = [np.zeros((npix, npix))] * N

        kwargs_maps = {'vmin': -vmin_max, 'vmax': vmin_max, 'cmap': 'bwr',
                       'origin': 'lower', 'extent': [-grid_size, grid_size, -grid_size, grid_size]}
        ax1.imshow(kappa_list[0] - kappa_target[0], **kwargs_maps)
        ax2.imshow(kappa_list[1] - kappa_target[1], **kwargs_maps)
        ax3.imshow(kappa_list[2] - kappa_target[2], **kwargs_maps)
        ax4.imshow(kappa_list[3] - kappa_target[3], **kwargs_maps)

    def plot_deflection_field(self, kwargs_solution_list, grid_size=0.2, grid_resolution=0.001, alpha_x_target=None,
                              alpha_y_target=None, vmin_max=0.05):

        npix = int(2 * grid_size / grid_resolution)
        _r = np.linspace(-grid_size, grid_size, npix)
        _xx, _yy = np.meshgrid(_r, _r)
        shape0 = _xx.shape
        alpha_x_list = []
        alpha_y_list = []
        for i, (xi, yi) in enumerate(zip(self._x_image_list, self._y_image_list)):
            alpha_x, alpha_y = self.lens_model.alpha(xi + _xx.ravel(), yi + _yy.ravel(), kwargs_solution_list[i])
            alpha_x_list.append(alpha_x.reshape(shape0))
            alpha_y_list.append(alpha_y.reshape(shape0))

        fig = plt.figure(1)
        fig.set_size_inches(16, 8)
        N = len(self._x_image_list)
        ax1 = plt.subplot(2, N, 1)
        ax2 = plt.subplot(2, N, 2)
        ax3 = plt.subplot(2, N, 3)
        ax4 = plt.subplot(2, N, 4)
        ax5 = plt.subplot(2, N, 5)
        ax6 = plt.subplot(2, N, 6)
        ax7 = plt.subplot(2, N, 7)
        ax8 = plt.subplot(2, N, 8)
        if alpha_x_target is None:
            alpha_x_target = [np.zeros((npix, npix))] * N
            alpha_y_target = [np.zeros((npix, npix))] * N

        kwargs_maps = {'vmin': -vmin_max, 'vmax': vmin_max, 'cmap': 'bwr',
                       'origin': 'lower', 'extent': [-grid_size, grid_size, -grid_size, grid_size]}
        ax1.imshow(alpha_x_list[0] - alpha_x_target[0], **kwargs_maps)
        ax2.imshow(alpha_x_list[1] - alpha_x_target[1], **kwargs_maps)
        ax3.imshow(alpha_x_list[2] - alpha_x_target[2], **kwargs_maps)
        ax4.imshow(alpha_x_list[3] - alpha_x_target[3], **kwargs_maps)
        ax5.imshow(alpha_y_list[0] - alpha_y_target[0], **kwargs_maps)
        ax6.imshow(alpha_y_list[1] - alpha_y_target[1], **kwargs_maps)
        ax7.imshow(alpha_y_list[2] - alpha_y_target[2], **kwargs_maps)
        ax8.imshow(alpha_y_list[3] - alpha_y_target[3], **kwargs_maps)

    def plot_images(self, source_fwhm_parsec, kwargs_solution_list, grid_size=None, grid_resolution=None):

        if grid_size is None:
            grid_size = auto_raytracing_grid_size(source_fwhm_parsec)
        if grid_resolution is None:
            grid_resolution = auto_raytracing_grid_resolution(source_fwhm_parsec)

        npix = int(2 * grid_size / grid_resolution)
        _r = np.linspace(-grid_size, grid_size, npix)
        _xx, _yy = np.meshgrid(_r, _r)
        shape0 = _xx.shape
        source_fwhm_arcsec = source_fwhm_parsec * self._cosmo.arcsec_per_kpc_proper(self._zsource).value/1000
        source_size_arcsec = source_fwhm_arcsec/(2 * np.sqrt(2 * np.log(2)))
        kwargs_source = [{'amp': 1.0, 'center_x': self._source_x, 'center_y': self._source_y, 'sigma': source_size_arcsec}]
        source_model = LightModel(['GAUSSIAN'])
        mags = []
        sb_list = []
        for i, (xi, yi) in enumerate(zip(self._x_image_list, self._y_image_list)):
            if self._keep_images[i] is False: continue
            beta_x, beta_y = self.lens_model.ray_shooting(xi + _xx.ravel(),
                                                          yi + _yy.ravel(),
                                                          kwargs_solution_list[i])
            sb = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
            mag = np.sum(sb) * grid_resolution ** 2
            mags.append(mag)
            sb_list.append(sb.reshape(shape0))

        fig = plt.figure(1)
        fig.set_size_inches(16, 6)
        N = len(sb_list)
        fr = np.array(mags)/mags[0]
        fr = fr/np.max(fr)
        for i in range(0, len(sb_list)):
            ax = plt.subplot(1, N, i + 1)
            ax.imshow(sb_list[i], origin='lower',
                      extent=[-grid_size, grid_size, -grid_size, grid_size])
            ax.annotate('magnification: ' + str(np.round(mags[i], 3)), xy=(0.05, 0.9), xycoords='axes fraction', color='w',
                        fontsize=12)
            ax.annotate('flux ratio: ' + str(np.round(fr[i], 3)), xy=(0.05, 0.8), xycoords='axes fraction', color='w',
                        fontsize=12)
        plt.show()

    def solve_kwargs_arc_from_kwargs(self, lens_model_estimate, kwargs_lens_estimate, angular_matching_scale,
                                     index_model_list=None, verbose=False):

        kwargs_full_list = []
        penalty_list = []
        if index_model_list is None:
            idx_min = 0
            idx_max = len(kwargs_lens_estimate)
            index_model_list = [[idx_min, idx_max]] * len(self.model_list)
        else:
            assert len(index_model_list) == len(self.model_list)

        for i, model in enumerate(self.model_list):
            if self._keep_images[i]:
                if verbose:
                    print('solving image '+str(i+1)+'... ')
                    t0 = time()
                _, kwargs_full, penalties = model.solve_kwargs_arc_from_kwargs(lens_model_estimate,
                                            kwargs_lens_estimate[index_model_list[i][0]:index_model_list[i][1]],
                                                                               angular_matching_scale)
                if verbose:
                    tellapsed = np.round((time() - t0)/60, 2)
                    print('finished image '+str(i+1)+' in '+str(tellapsed)+' min.')
                    print('curved arc fit penalties: ', penalties)

                kwargs_full_list.append(kwargs_full)
                penalty_list.append(penalties)
        return kwargs_full_list, penalty_list

    def solve_kwargs_arc(self, kwargs_arc_constraint_list, angular_matching_scale, radial_stretch_sigma=1e-9,
                         tangential_stretch_sigma=1e-9, curvature_sigma=1e-9, direction_sigma=1e-9,
                         tan_diff_sigma=1e-9):

        kwargs_full_list = []
        penalty_list = []
        for i, model in enumerate(self.model_list):
            _, kwargs_full, penalties = model.solve_kwargs_arc(kwargs_arc_constraint_list[i],
                                                               angular_matching_scale, radial_stretch_sigma,
                                                               tangential_stretch_sigma, curvature_sigma,
                                                               direction_sigma, tan_diff_sigma)
            kwargs_full_list.append(kwargs_full)
            penalty_list.append(penalties)
        return kwargs_full_list, penalty_list

    def mags(self, kwargs_model_list, source_fwhm_pc):

        mags = []
        for i, (xi, yi) in enumerate(zip(self._x_image_list, self._y_image_list)):
            if self._keep_images[i]:

                if source_fwhm_pc == 0:
                    # assume point source
                    m = abs(self.lens_model_extension._lensModel.magnification(xi, yi, kwargs_model_list[i]))
                else:
                    m = self.lens_model_extension.magnification_finite_adaptive([xi], [yi], self._source_x, self._source_y,
                                                                           kwargs_model_list[i], source_fwhm_pc, self._zsource,
                                                                           cosmo=self._cosmo)
                mags.append(float(m))

        if np.sum(self._keep_images)==1:
            return float(np.array(mags))
        else:
            return np.array(mags)


class MultiScaleModel(object):
    """
    This class uses either a first order distortion (HESSIAN) or a non-linear approximation of local lensing properties
    (CURVED_ARC) to match the convergence and shear on an arbitrary angular scale. The constraints on the convergence and
    shear could come, for example, from a smooth lens profile fit to an extended image.

    The code outputs the keyword arguments for either kwargs_hessian or kwargs_arc such that the full
    lens system has the same kappa/gamma values on the desired angular scale.

    For single plane lensing, this is a linear problem because kappa_1 = kappa_2 + kappa_3 (same for shear components).

    For multi-plane lensing, however, it is nonlinear because the function to be optimized (HESSIAN or CURVED_ARC)
    appears inside the argument of the deflection angle for lens models situated between the main lens
    plane and the source.
    """

    def __init__(self, x_image, y_image,
                 source_x, source_y,
                 lens_model_list_other, redshift_list_other,
                 kwargs_lens_other, z_lens, z_source, cosmo, tan_diff=False, finite_area=False):

        """
        We initialize the class by specifying the image coordinate, and
        lens_model_list_other/redshift_list_other/kwargs_lens_other. These refer to any lens models besides
        the macromodel (e.g. dark matter halos).

        :param x_image: x image position in arcsec
        :param y_image: y image position in arcsec
        :param source_x: source x position in arcsec
        :param source_y: source y position in arcsec
        :param lens_model_list_other: a list of other lens models to be included in the computation
        (could be dark matter halos)
        :param redshift_list_other: a list of redshifts for the other lens models
        :param kwargs_lens_other: list of kwargs for other lens models
        :param z_lens: the main deflector redshift; this is the redshift where CURVED_ARC lives
        :param z_source: source redshift
        :param cosmo: an instance of astropy
        :param tan_diff: whether to use the information in the differential of the tangential stretch
        :param finite_area: whether to perform curved arc estimate over finite area dr
        """

        self.cosmo = cosmo
        self._tan_diff = tan_diff
        if finite_area and tan_diff:
            raise Exception('cannot specify both finite_area=True and tan_diff=True')
        self._finite_area = finite_area
        self._x = x_image
        self._y = y_image
        self._zlens = z_lens
        self._zsource = z_source

        self._lens_model_list_other = lens_model_list_other
        self._redshift_list_other = list(redshift_list_other)
        self._kwargs_lens_other = kwargs_lens_other
        self._source_x = source_x
        self._source_y = source_y

        self._kwargs_shift = None
        self.compute_kwargs_shift()

        self._lens_model_other = LensModel(self._lens_model_list_other, lens_redshift_list=self._redshift_list_other,
                                           z_source=self._zsource, multi_plane=True, cosmo=cosmo)

        self.kwargs_init = {'x_image': self._x, 'y_image': self._y,
                            'source_x': source_x, 'source_y': source_y,
                            'lens_model_list_other': lens_model_list_other,
                            'redshift_list_other': redshift_list_other,
                            'kwargs_lens_other': kwargs_lens_other,
                            'z_lens': z_lens, 'z_source': z_source, 'cosmo': cosmo}

        self._kwargs_curved_arc_estimate = {}
        if self._tan_diff:
            raise Exception('not yet implemented')
            # self._curved_arc_type = ['CURVED_ARC_TAN_DIFF', 'SHIFT']
            # self._param_class = CurvedArcTanDiff
            # self._kwargs_curved_arc_estimate['tan_diff'] = True
        else:
            self._curved_arc_type = ['CURVED_ARC_SPP', 'SHIFT']
            self._param_class = CurvedArc

        self._multiplane_fast_differential = None
        self.lens_model = LensModel(self._curved_arc_type + self._lens_model_list_other,
                                    lens_redshift_list=[z_lens]*2 + self._redshift_list_other,
                                    z_source=self._zsource, multi_plane=True, cosmo=cosmo)
        self.extension = LensModelExtensions(self.lens_model)
        if self._finite_area:
            raise Exception('not yet implemented')
            #self._curved_arc_estimate_function = self.extension.curved_arc_finite_area
        # else:
        #     self._curved_arc_estimate_function = self.extension.curved_arc_estimate

        """
        The following lines are necessary because we have to place the CURVED_ARC model at the angular position where
        a light ray interests the main lens plane. Due to foreground lensing, this is not necessarily the
        observed coordinate.
        """
        x, y, _, _ = self._lens_model_other.lens_model.ray_shooting_partial(
            0., 0., x_image, y_image, 0., self._zlens, self._kwargs_lens_other)
        d = self.cosmo.comoving_transverse_distance(self._zlens).value
        self._img_plane_x = x / d
        self._img_plane_y = y / d
        self.set_minimization_function()

    def set_minimization_function(self, function='nelder-mead'):

        self._minimize_method = function

    def lensmodel_shift(self):

        lens_model_list = ['SHIFT'] + self._lens_model_list_other
        redshift_list = [self._zlens] + self._redshift_list_other
        lens_model = LensModel(lens_model_list, lens_redshift_list=redshift_list,
                               multi_plane=True, z_source=self._zsource, cosmo=self.cosmo)
        kwargs_shift = self.compute_kwargs_shift()
        kwargs = [kwargs_shift] + self._kwargs_lens_other
        return lens_model, kwargs

    def compute_kwargs_shift(self):

        """
        Solve for deflection angles that map each observed image coordinate to the source position
        :return: kwargs_shift
        """

        if self._kwargs_shift is None:
            lens_model_list = ['SHIFT'] + self._lens_model_list_other
            redshift_list = [self._zlens] + self._redshift_list_other
            kwargs_lens = [{'alpha_x': 0., 'alpha_y': 0.}] + self._kwargs_lens_other
            param_class_shift = ShiftLensModelParamManager(kwargs_lens)
            fast_ray_shooting = MultiplaneFast(self._x, self._y, self._zlens, self._zsource, lens_model_list, redshift_list,
                 self.cosmo, param_class_shift, None)
            x0 = np.array([self._x, self._y])
            opt = minimize(self._coordinate_func_to_min, x0, method='Nelder-Mead',
                           args=(self._source_x, self._source_y, fast_ray_shooting))['x']
            kwargs = param_class_shift.args_to_kwargs(opt)
            self._kwargs_shift = kwargs[0]

        return self._kwargs_shift

    def solve_kwargs_arc_from_kwargs(self, lens_model_estimate, kwargs_lens_estimate, angular_matching_scale,
                                     radial_stretch_sigma=1e-9,
                                     tangential_stretch_sigma=1e-9, curvature_sigma=1e-9, direction_sigma=1e-9,
                                     tan_diff_sigma=1e-9):

        if lens_model_estimate.lens_model_list[0] in ['CURVED_ARC_SPP', 'CURVED_ARC_TAN_DIFF']:
            # here we have a model fit to data that is itself a curved arc, so we just use these parameters
            assert lens_model_estimate.lens_model_list[1] == 'SHIFT' # make sure the second lens model is a shift
            kwargs_curved_arc = kwargs_lens_estimate[0]
            return self.solve_kwargs_arc(kwargs_curved_arc, angular_matching_scale, radial_stretch_sigma,
                                         tangential_stretch_sigma, curvature_sigma, direction_sigma, tan_diff_sigma)
        #else:

        ext = LensModelExtensions(lens_model_estimate)
        if self._finite_area:
            kwargs_curved_arc = ext.curved_arc_finite_area(self._x, self._y, kwargs_lens_estimate,
                                                           dr=angular_matching_scale)
        else:
            kwargs_curved_arc = ext.curved_arc_estimate(self._x, self._y, kwargs_lens_estimate,
                                                    smoothing=angular_matching_scale,
                                                    smoothing_3rd=angular_matching_scale, tan_diff=self._tan_diff)

        return self.solve_kwargs_arc(kwargs_curved_arc, angular_matching_scale, radial_stretch_sigma,
                         tangential_stretch_sigma, curvature_sigma, direction_sigma, tan_diff_sigma)

    def solve_kwargs_arc(self, kwargs_arc_constraint, angular_matching_scale, radial_stretch_sigma=1e-9,
                         tangential_stretch_sigma=1e-9, curvature_sigma=1e-9, direction_sigma=1e-9, tan_diff_sigma=1e-9):

        """

        :kwargs_arc_constraint: the target parameters of the curved arc model
        :param angular_matching_scale: the angular scale used to compute the finite-difference derivatives in the hessian
        :return: kwargs that yield the specified kappa/gamma values on the angular scale angular_matching_scale
        """

        kwargs_shift = self.compute_kwargs_shift()
        kwargs_arc_constraint['center_x'] = self._img_plane_x
        kwargs_arc_constraint['center_y'] = self._img_plane_y
        kwargs_lens = [kwargs_arc_constraint] + [kwargs_shift] + self._kwargs_lens_other
        param_class_curved_arc = self._param_class(kwargs_lens)
        initial_guess = param_class_curved_arc.kwargs_to_args([kwargs_arc_constraint])
        if self._finite_area:
            raise Exception('not yet implemented')
            self._kwargs_curved_arc_estimate['dr'] = angular_matching_scale
            self._kwargs_curved_arc_estimate['num_points'] = 5
        else:
            self._kwargs_curved_arc_estimate['smoothing'] = angular_matching_scale
            self._kwargs_curved_arc_estimate['smoothing_3rd'] = angular_matching_scale

        self._multiplane_fast_differential = MultiplaneFastDifferential(self._x, self._y, self._zlens, self._zsource,
                                                                        self.lens_model,
                                                                        self._curved_arc_type + self._lens_model_list_other,
                                                                        [self._zlens] * 2 + self._redshift_list_other,
                                                                        self.cosmo, angular_matching_scale,
                                                                        param_class_curved_arc)

        if self._tan_diff:
            (radial_stretch_target, tangential_stretch_target, curvature_target, direction_target, dtan_dtan_target) = \
                param_class_curved_arc.kwargs_to_args([kwargs_arc_constraint])
            args_minimize = (radial_stretch_target,
                             radial_stretch_sigma,
                             tangential_stretch_target,
                             tangential_stretch_sigma,
                             curvature_target,
                             curvature_sigma,
                             direction_target,
                             direction_sigma,
                             dtan_dtan_target,
                             tan_diff_sigma,
                             param_class_curved_arc)
            minimize_func = self._model_arc_tan_diff

        else:
            (radial_stretch_target, tangential_stretch_target, curvature_target, direction_target) = \
            param_class_curved_arc.kwargs_to_args([kwargs_arc_constraint])
            args_minimize = (radial_stretch_target,
                             radial_stretch_sigma,
                             tangential_stretch_target,
                             tangential_stretch_sigma,
                             curvature_target,
                             curvature_sigma,
                             direction_target,
                             direction_sigma,
                             param_class_curved_arc)
            minimize_func = self._model_arc_spp

        opt = minimize(minimize_func, initial_guess, method=self._minimize_method,
                       args=args_minimize, options={'adaptive': True, 'maxiter': 6000})
        kwargs_full = param_class_curved_arc.args_to_kwargs(opt['x'])
        kwargs_curved_arc = kwargs_full[0]
        if self._tan_diff:
            penalty_final = self._sorted_penalties_tan_diff(opt['x'], *args_minimize)
        else:
            penalty_final = self._sorted_penalties_spp(opt['x'], *args_minimize)
        return kwargs_curved_arc, kwargs_full, penalty_final

    @staticmethod
    def _coordinate_func_to_min(args, source_x, source_y, fast_ray_shooting):

        """
        Used to compute alpha_shift kwargs
        :param args:
        :param source_x:
        :param source_y:
        :param fast_ray_shooting:
        :return:
        """
        betax, betay = fast_ray_shooting.ray_shooting_fast(args)
        dx = abs(betax - source_x)
        dy = abs(betay - source_y)
        return (dx + dy) / 0.00001

    def _model_arc_spp(self, args,
                   radial_stretch_target,
                   radial_stretch_sigma,
                   tangential_stretch_target,
                   tangential_stretch_sigma,
                   curvature_target,
                   curvature_sigma,
                   direction_target,
                   direction_sigma,
                   param_class_curved_arc):

        penalities = self._sorted_penalties_spp(args,
                                                     radial_stretch_target,
                                                     radial_stretch_sigma,
                                                     tangential_stretch_target,
                                                     tangential_stretch_sigma,
                                                     curvature_target,
                                                     curvature_sigma,
                                                     direction_target,
                                                     direction_sigma,
                                                     param_class_curved_arc)
        return np.sum(penalities)

    def _model_arc_tan_diff(self, args,
                   radial_stretch_target,
                   radial_stretch_sigma,
                   tangential_stretch_target,
                   tangential_stretch_sigma,
                   curvature_target,
                   curvature_sigma,
                   direction_target,
                   direction_sigma,
                    dtan_dtan_target,
                    tan_diff_sigma,
                   param_class_curved_arc):

        penalities = self._sorted_penalties_tan_diff(args,
                            radial_stretch_target,
                            radial_stretch_sigma,
                            tangential_stretch_target,
                            tangential_stretch_sigma,
                            curvature_target,
                            curvature_sigma,
                            direction_target,
                            direction_sigma,
                            dtan_dtan_target,
                            tan_diff_sigma,
                            param_class_curved_arc)
        return np.sum(penalities)

    def _sorted_penalties_tan_diff(self, args,
                            radial_stretch_target,
                            radial_stretch_sigma,
                            tangential_stretch_target,
                            tangential_stretch_sigma,
                            curvature_target,
                            curvature_sigma,
                            direction_target,
                            direction_sigma,
                            dtan_dtan_target,
                            tan_diff_sigma,
                            param_class_curved_arc):

        kwargs_lens = param_class_curved_arc.args_to_kwargs(args)
        model_estimate = self._curved_arc_estimate_function(self._x, self._y, kwargs_lens,
                                                            **self._kwargs_curved_arc_estimate)

        (radial_stretch, tangential_stretch, curvature, direction, dtan_dtan) = param_class_curved_arc.kwargs_to_args(
            [model_estimate])
        penalty_radial_stretch = 0.0
        penalty_tangential_stretch = 0.0
        penalty_curvature = 0.0
        penalty_direction = 0.0
        penalty_tan_diff = 0.0
        if radial_stretch_sigma > 0:
            penalty_radial_stretch = (radial_stretch - radial_stretch_target) ** 2 / radial_stretch_sigma ** 2
        if tangential_stretch_sigma > 0:
            penalty_tangential_stretch += (tangential_stretch - tangential_stretch_target) ** 2 / tangential_stretch_sigma ** 2
        if curvature_sigma > 0:
            penalty_curvature += (curvature - curvature_target) ** 2 / curvature_sigma ** 2
        if direction_sigma > 0:
            penalty_direction += (direction - direction_target) ** 2 / direction_sigma ** 2
        if tan_diff_sigma > 0:
            penalty_tan_diff += (dtan_dtan - dtan_dtan_target) ** 2 / tan_diff_sigma ** 2
        return [penalty_radial_stretch, penalty_tangential_stretch, penalty_curvature,
                penalty_direction, penalty_tan_diff]

    def _sorted_penalties_spp(self, args,
                   radial_stretch_target,
                   radial_stretch_sigma,
                   tangential_stretch_target,
                   tangential_stretch_sigma,
                   curvature_target,
                   curvature_sigma,
                   direction_target,
                   direction_sigma,
                   param_class_curved_arc):

        # kwargs_lens = param_class_curved_arc.args_to_kwargs(args)
        # model_estimate = self._curved_arc_estimate_function(self._x, self._y, kwargs_lens,
        #                                                     **self._kwargs_curved_arc_estimate)
        # (radial_stretch, tangential_stretch, curvature, direction) = param_class_curved_arc.kwargs_to_args(
        #     [model_estimate])
        kwargs = self._multiplane_fast_differential.\
            curved_arc_estimate_fast(args)
        (radial_stretch, tangential_stretch, curvature, direction) = param_class_curved_arc.kwargs_to_args([kwargs])
        penalty_radial_stretch = 0.0
        penalty_tangential_stretch = 0.0
        penalty_curvature = 0.0
        penalty_direction = 0.0
        if radial_stretch_sigma > 0:
            penalty_radial_stretch = (radial_stretch - radial_stretch_target) ** 2 / radial_stretch_sigma ** 2
        if tangential_stretch_sigma > 0:
            penalty_tangential_stretch += (tangential_stretch - tangential_stretch_target) ** 2 / tangential_stretch_sigma ** 2
        if curvature_sigma > 0:
            penalty_curvature += (curvature - curvature_target) ** 2 / curvature_sigma ** 2
        if direction_sigma > 0:
            penalty_direction += (direction - direction_target) ** 2 / direction_sigma ** 2
        return [penalty_radial_stretch, penalty_tangential_stretch, penalty_curvature, penalty_direction]

def filter_realization(x_images, y_images, source_x, source_y, zsource, realization, models,
                       aperture_radius=0.25, log_mass_in_aperture=0.0,
                       log_mass_global=7.7):

    ray_x_interp, ray_y_interp = [], []
    for i, model in enumerate(models):
        lens_model_shift, kwargs_lens = model.lensmodel_shift()
        ray_x, ray_y, _ = interpolate_ray_paths([x_images[i]], [y_images[i]], lens_model_shift, kwargs_lens,
                                                zsource, terminate_at_source=True,
                                                source_x=source_x, source_y=source_y)
        ray_x_interp += ray_x
        ray_y_interp += ray_y

    realization_filtered = realization.filter(aperture_radius, aperture_radius,
                                              log_mass_in_aperture,
                                              log_mass_in_aperture,
                                              log_mass_global,
                                              log_mass_global,
                                              ray_x_interp, ray_y_interp)

    return realization_filtered

def align_realization(x_images, y_images, source_x, source_y, zsource, realization, models):

    ray_interp_x = []
    ray_interp_y = []
    for i, model in enumerate(models):
        lens_model_shift, kwargs_init = model.lensmodel_shift()
        _ray_interp_x, _ray_interp_y, distances = interpolate_ray_paths([x_images[i]], [y_images[i]],
                                                                        lens_model_shift,
                                                                        kwargs_init, zsource,
                                                                        terminate_at_source=True,
                                                                        source_x=source_x,
                                                                        source_y=source_y)
        ray_interp_x.append(_ray_interp_x[0])
        ray_interp_y.append(_ray_interp_y[0])
    ### Now compute the centroid of the light cone as the coordinate centroid of the individual images
    angular_coordinates_x = []
    angular_coordinates_y = []
    for di in distances:
        x_coords = [ray_x(di) for ray_x in ray_interp_x]
        y_coords = [ray_y(di) for ray_y in ray_interp_y]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        angular_coordinates_x.append(x_center)
        angular_coordinates_y.append(y_center)
    ray_interp_x_center = interp1d(distances, angular_coordinates_x)
    ray_interp_y_center = interp1d(distances, angular_coordinates_y)
    realization_shifted = realization.shift_background_to_source(ray_interp_x_center, ray_interp_y_center)
    return realization_shifted

    # def solve_kwargs_arc_constraint_kappagamma(self, kwargs_curved_arc_init, kappa_constraint,
    #                                            gamma_constraint1, gamma_constraint2, angular_matching_scale,
    #                                            fit_setting='FIXED_CURVATURE_DIRECTION'):
    #
    #     """
    #
    #     :param kwargs_curved_arc_init: an initial guess for the curved_arc parameters
    #     :param kappa_constraint: the constraint on the convergence to match
    #     :param gamma_constraint1: the constraint on the 1st shear component to match
    #     gamma1 = 0.5 * (fxx - fyy)
    #     :param gamma_constraint2: the constraint on the 2nd shear component to match
    #     I have defined this as gamma2 = 0.5 * (fxy + fyx) since fxy != fyx in multi-plane lensing
    #     :param angular_matching_scale: the angular scale used to compute the finite-difference derivatives in the hessian
    #     :param fit_setting: a string that specifies a param class (see param_mangaer.py) to fix certain constraints
    #     on the curved_arc model. Options are 'FIXED_CURVATURE_DIRECTION', 'FIXED_CURVATURE'
    #     :return: kwargs that yield the specified kappa/gamma values on the angular scale angular_matching_scale
    #     """
    #
    #     kwargs_shift = self.compute_kwargs_shift()
    #
    #     lens_model_list = ['CURVED_ARC_TAN_DIFF', 'SHIFT'] + self._lens_model_list_other
    #
    #     # we have to make sure the center of the curved arc model is at the coordinate where
    #     # the ray hits the image plane, not necessarily = to the observed image position!
    #     kwargs_curved_arc_init['center_x'] = self._img_plane_x
    #     kwargs_curved_arc_init['center_y'] = self._img_plane_y
    #
    #     kwargs_lens = [kwargs_curved_arc_init] + [kwargs_shift] + self._kwargs_lens_other
    #
    #     redshift_list = [self._zlens] * 2 + self._redshift_list_other
    #
    #     if fit_setting == 'FIXED_CURVATURE_DIRECTION':
    #         param_class_curved_arc = CurvedArcFixedCurvatureDirection(kwargs_lens)
    #     elif fit_setting == 'FIXED_CURVATURE':
    #         param_class_curved_arc = CurvedArcFixedCurvature(kwargs_lens)
    #     elif fit_setting == 'FREE':
    #         param_class_curved_arc = CurvedArcFree(kwargs_lens)
    #     else:
    #         raise Exception('fit_setting ' + str(fit_setting) + ' not recognized')
    #
    #     fast_ray_shooting = MultiplaneFastDifferential(angular_matching_scale, self._x, self._y, self._img_plane_x,
    #                                                    self._img_plane_y,
    #                                                    self._zlens, self._zsource,
    #                                                    lens_model_list, redshift_list, self._astropy,
    #                                                    param_class_curved_arc)
    #
    #     args_minimize = (kappa_constraint, gamma_constraint1, gamma_constraint2, fast_ray_shooting,
    #                      param_class_curved_arc)
    #
    #     x0 = param_class_curved_arc.kwargs_to_args([kwargs_curved_arc_init])
    #
    #     opt = minimize(self._func_to_min_kappagamma_constraint, x0, method='Nelder-Mead',
    #                    args=args_minimize, options={'adaptive': True})['x']
    #
    #     kwargs_full = param_class_curved_arc.args_to_kwargs(opt)
    #     kwargs_curved_arc = kwargs_full[0]
    #
    #     result = self._kappa_gamma(opt, fast_ray_shooting)
    #
    #     return kwargs_curved_arc, kwargs_full, result
    #
    # def _func_to_min_kappagamma_constraint(self, args, kappa, gamma1, gamma2, fast_ray_shooting, param_class):
    #
    #     """
    #     used to match the hessian constraints
    #     :param args:
    #     :param kappa:
    #     :param gamma1:
    #     :param gamma2:
    #     :param fast_ray_shooting:
    #     :param param_class:
    #     :return:
    #     """
    #     arg_penalty = param_class.param_chi_square_penalty(args)
    #
    #     kappa_model, gamma_model_1, gamma_model_2 = self._kappa_gamma(args, fast_ray_shooting)
    #
    #     dx = kappa - kappa_model
    #     dy = gamma_model_1 - gamma1
    #     dz = gamma_model_2 - gamma2
    #
    #     penalty = (dx ** 2 + dy ** 2 + dz ** 2) / 0.00001 ** 2 + arg_penalty
    #
    #     return penalty

# def solve_kwargs_hessian(self, hessian_init, kappa_constraint,
    #                          gamma_constraint1, gamma_constraint2, angular_matching_scale):
    #
    #     """
    #
    #     :param hessian_init: an initial guess for the hessian
    #     :param kappa_constraint: the constraint on the convergence to match
    #     :param gamma_constraint1: the constraint on the 1st shear component to match
    #     gamma1 = 0.5 * (fxx - fyy)
    #     :param gamma_constraint2: the constraint on the 2nd shear component to match
    #     I have defined this as gamma2 = 0.5 * (fxy + fyx) since fxy != fyx in multi-plane lensing
    #     :param angular_matching_scale: the angular scale used to compute the finite-difference derivatives in the hessian
    #     :return: kwargs that yield the specified kappa/gamma values on the angular scale angular_matching_scale
    #     """
    #
    #     hessian_init['ra_0'] = self._img_plane_x
    #     hessian_init['dec_0'] = self._img_plane_y
    #     kwargs_shift = self.compute_kwargs_shift()
    #     lens_model_list = ['HESSIAN', 'SHIFT'] + self._lens_model_list_other
    #     kwargs_lens = [hessian_init] + [kwargs_shift] + self._kwargs_lens_other
    #     redshift_list = [self._zlens] * 2 + self._redshift_list_other
    #     param_class_hessian = HessianEqualPartials(kwargs_lens)
    #     fast_ray_shooting = MultiplaneFastDifferential(angular_matching_scale, self._x, self._y, self._img_plane_x,
    #                                                    self._img_plane_y,
    #                                                    self._zlens, self._zsource,
    #                                                    lens_model_list, redshift_list, self._astropy,
    #                                                    param_class_hessian)
    #     self._fast_ray_shooting_differential = fast_ray_shooting
    #     x0 = param_class_hessian.kwargs_to_args([hessian_init])
    #     args_minimize = (kappa_constraint, gamma_constraint1, gamma_constraint2, fast_ray_shooting,
    #                      param_class_hessian)
    #     opt = minimize(self._func_to_min_kappagamma_constraint, x0, method='Nelder-Mead',
    #                    args=args_minimize)['x']
    #     kwargs_full = param_class_hessian.args_to_kwargs(opt)
    #     kwargs_hessian = kwargs_full[0]
    #     result = self._kappa_gamma(opt, fast_ray_shooting)
    #     return kwargs_hessian, kwargs_full, result

    # @staticmethod
    # def _kappa_gamma(args, fast_ray_shooting):
    #
    #     """
    #     computes kappa and gamma from the hessian components
    #     since fxy != fyx for multi-plane systems I define gamma2 as the mean between them
    #     :param args:
    #     :param fast_ray_shooting:
    #     :return:
    #     """
    #     fxx, fxy, fyx, fyy = fast_ray_shooting.hessian(args)
    #
    #     kappa_model = 0.5 * (fxx + fyy)
    #     gamma_model_1 = 0.5 * (fxx - fyy)
    #     gamma_model_2 = 0.5 * (fxy + fyx)
    #     return kappa_model, gamma_model_1, gamma_model_2
