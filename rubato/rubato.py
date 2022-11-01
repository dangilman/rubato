import numpy as np
from scipy.optimize import minimize
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.lens_model import LensModel
from rubato.util import MultiplaneFast, MultiplaneFastDifferential
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from rubato.param_manager import ShiftLensModelParamManager, CurvedArcFree


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

    def __init__(self, x_image_coordinate, y_image_coordinate,
                 source_x_coordinate, source_y_coordinate,
                 lens_model_list_other, redshift_list_other,
                 kwargs_lens_other, z_lens, z_source, astropy_instance=None):

        """
        We initialize the class by specifying the image coordinate, and
        lens_model_list_other/redshift_list_other/kwargs_lens_other. These refer to any lens models besides
        the macromodel (e.g. dark matter halos).

        :param x_image_coordinate: x image position in arcsec
        :param y_image_coordinate: y image position in arcsec
        :param source_x_coordinate: source x position in arcsec
        :param source_y_coordinate: source y position in arcsec
        :param lens_model_list_other: a list of other lens models to be included in the computation
        (could be dark matter halos)
        :param redshift_list_other: a list of redshifts for the other lens models
        :param kwargs_lens_other: list of kwargs for other lens models
        :param z_lens: the main deflector redshift; this is the redshift where CURVED_ARC lives
        :param z_source: source redshift
        :param astropy_instance: an instance of astropy
        """

        if astropy_instance is None:
            background = Background()
            astropy_instance = background.cosmo

        self._astropy = astropy_instance

        self._x = x_image_coordinate
        self._y = y_image_coordinate
        self._zlens = z_lens
        self._zsource = z_source

        self._lens_model_list_other = lens_model_list_other
        self._redshift_list_other = list(redshift_list_other)
        self._kwargs_lens_other = kwargs_lens_other
        self._source_x = source_x_coordinate
        self._source_y = source_y_coordinate

        self._kwargs_shift = None
        self.compute_kwargs_shift()

        self._lens_model_other = LensModel(self._lens_model_list_other, lens_redshift_list=self._redshift_list_other,
                                           z_source=self._zsource, multi_plane=True)

        """
        The followinig lines are necessary because we have to place the CURVED_ARC model at the angular position where
        a light ray interests the main lens plane. Due to foreground lensing, this is not necessarily the
        observed coordinate.
        """
        self.cosmo = self._lens_model_other.lens_model.cosmo
        x, y, _, _ = self._lens_model_other.lens_model.ray_shooting_partial(
            0., 0., x_image_coordinate, y_image_coordinate, 0., self._zlens, self._kwargs_lens_other)
        d = self.cosmo.comoving_transverse_distance(self._zlens).value
        self._img_plane_x = x / d
        self._img_plane_y = y / d

    def image_flux(self, kwargs_curved_arc, source_model, kwargs_source, npix, grid_radius_arcsec):

        kwargs_shift = self.compute_kwargs_shift()
        lens_model_list = ['CURVED_ARC_SPP', 'SHIFT'] + self._lens_model_list_other
        kwargs_lens = [kwargs_curved_arc] + [kwargs_shift] + self._kwargs_lens_other
        redshift_list = [self._zlens] * 2 + self._redshift_list_other
        lens_model = LensModel(lens_model_list, z_source=self._zsource, multi_plane=True,
                               lens_redshift_list=redshift_list, cosmo=self._astropy)
        _r = np.linspace(-grid_radius_arcsec, grid_radius_arcsec, npix)
        _xx, _yy = np.meshgrid(_r, _r)
        beta_x, beta_y = lens_model.ray_shooting(_xx.ravel() + self._x,
                                                 _yy.ravel() + self._y,
                                                 kwargs_lens)
        sb = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
        return float(np.sum(sb) * (2*grid_radius_arcsec/npix)**2)

    @staticmethod
    def _kappa_gamma(args, fast_ray_shooting):

        """
        computes kappa and gamma from the hessian components
        since fxy != fyx for multi-plane systems I define gamma2 as the mean between them
        :param args:
        :param fast_ray_shooting:
        :return:
        """
        fxx, fxy, fyx, fyy = fast_ray_shooting.hessian(args)

        kappa_model = 0.5 * (fxx + fyy)
        gamma_model_1 = 0.5 * (fxx - fyy)
        gamma_model_2 = 0.5 * (fxy + fyx)
        return kappa_model, gamma_model_1, gamma_model_2

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
            fast_ray_shooting = MultiplaneFast.fromModelList(self._x, self._y, self._zlens, self._zsource,
                                                             lens_model_list, redshift_list, self._astropy,
                                                             param_class_shift, None)

            x0 = np.array([self._x, self._y])
            opt = minimize(self._coordinate_func_to_min, x0, method='Nelder-Mead',
                           args=(self._source_x, self._source_y, fast_ray_shooting))['x']

            kwargs = param_class_shift.args_to_kwargs(opt)

            self._kwargs_shift = kwargs[0]

        return self._kwargs_shift

    def solve_kwargs_arc_constraint_curved_arc(self, kwargs_arc_constraint,
                                               angular_matching_scale,
                                               radial_stretch_sigma=0.01, tangential_stretch_sigma=0.01,
                                               curvature_sigma=0.01, direction_sigma=0.005):

        """

        :kwargs_arc_constraint: the target parameters of the curved arc model
        :param angular_matching_scale: the angular scale used to compute the finite-difference derivatives in the hessian
        :return: kwargs that yield the specified kappa/gamma values on the angular scale angular_matching_scale
        """

        kwargs_shift = self.compute_kwargs_shift()
        lens_model_list = ['CURVED_ARC_SPP', 'SHIFT'] + self._lens_model_list_other
        kwargs_arc_constraint['center_x'] = self._img_plane_x
        kwargs_arc_constraint['center_y'] = self._img_plane_y
        kwargs_lens = [kwargs_arc_constraint] + [kwargs_shift] + self._kwargs_lens_other
        redshift_list = [self._zlens] * 2 + self._redshift_list_other

        param_class_curved_arc = CurvedArcFree(kwargs_lens)
        fast_ray_shooting = MultiplaneFastDifferential(angular_matching_scale, self._x, self._y, self._img_plane_x,
                                                       self._img_plane_y,
                                                       self._zlens, self._zsource,
                                                       lens_model_list, redshift_list, self._astropy,
                                                       param_class_curved_arc)

        x0 = param_class_curved_arc.kwargs_to_args([kwargs_arc_constraint])
        (radial_stretch_target, tangential_stretch_target, curvature_target, direction_target) = x0
        args_minimize = (radial_stretch_target,
                         radial_stretch_sigma,
                         tangential_stretch_target,
                         tangential_stretch_sigma,
                         curvature_target,
                         curvature_sigma,
                         direction_target,
                         direction_sigma,
                         fast_ray_shooting)

        opt = minimize(self._func_to_min_arc_constraint, x0, method='Nelder-Mead',
                       args=args_minimize, options={'adaptive': True})['x']

        kwargs_full = param_class_curved_arc.args_to_kwargs(opt)
        kwargs_curved_arc = kwargs_full[0]
        result = opt
        return kwargs_curved_arc, kwargs_full, result

    def _func_to_min_arc_constraint(self, args,
                                    radial_stretch_target,
                                    radial_stretch_sigma,
                                    tangential_stretch_target,
                                    tangential_stretch_sigma,
                                    curvature_target,
                                    curvature_sigma,
                                    direction_target,
                                    direction_sigma,
                                    fast_ray_shooting):

        # evaluate the model with curved arc parameters plus on the specified angular scale
        args_model = fast_ray_shooting.curved_arc_estimate_fast(args)
        (radial_stretch, tangential_stretch, curvature, direction) = args_model

        penalty = 0
        penalty += (radial_stretch - radial_stretch_target) ** 2 / radial_stretch_sigma ** 2
        penalty += (tangential_stretch - tangential_stretch_target) ** 2 / tangential_stretch_sigma ** 2
        penalty += (curvature - curvature_target) ** 2 / curvature_sigma ** 2
        penalty += (direction - direction_target) ** 2 / direction_sigma ** 2
        return np.sqrt(penalty)

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
