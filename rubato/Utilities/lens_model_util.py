from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.QuadOptimizer.multi_plane_fast import MultiplaneFast

class MultiplaneFastDifferential(object):
    """
    This class uses the ray tracing routines in MultiPlaneFast to compute numerical derivatives of deflection angles
    i.e. the components of the hessian matrix
    """

    def __init__(self, x, y, z_lens, z_source, lens_model, lens_model_list, redshift_list, astropy_instance,
                 diff, param_class, tan_diff=False):

        self._x, self._y = x, y

        self._model_ra_pp = MultiplaneFast(x + diff/2, y + diff/2, z_lens, z_source, lens_model_list, redshift_list, astropy_instance,
                                                   param_class, None)
        self._model_ra_pn = MultiplaneFast(x + diff / 2, y - diff / 2, z_lens, z_source, lens_model_list, redshift_list,
                                           astropy_instance, param_class, None)
        self._model_ra_np = MultiplaneFast(x - diff / 2, y + diff / 2, z_lens, z_source, lens_model_list, redshift_list,
                                           astropy_instance, param_class, None)
        self._model_ra_nn = MultiplaneFast(x - diff / 2, y - diff / 2, z_lens, z_source, lens_model_list, redshift_list,
                                           astropy_instance, param_class, None)
        self._tan_diff = tan_diff
        self._diff = diff
        self.extension = LensModelExtensions(lens_model)
        self._param_class = param_class

    def curved_arc_estimate_fast(self, args):
        """
        performs the estimation of the curved arc description at a particular position of an arbitrary lens profile

        :param x: float, x-position where the estimate is provided
        :param y: float, y-position where the estimate is provided
        :param kwargs_lens: lens model keyword arguments
        :param smoothing: (optional) finite differential of second derivative (radial and tangential stretches)
        :param smoothing_3rd: differential scale for third derivative to estimate the tangential curvature
        :param tan_diff: boolean, if True, also returns the relative tangential stretch differential in tangential direction
        :return: keyword argument list corresponding to a CURVED_ARC profile at (x, y) given the initial lens model
        """
        kwargs_lens = self._param_class.args_to_kwargs(args)
        radial_stretch, tangential_stretch, v_rad1, v_rad2, v_tang1, v_tang2 = self.radial_tangential_stretch_fast(args)
        dx_tang = self._x + self._diff * v_tang1
        dy_tang = self._y + self._diff * v_tang2
        _, _, _, _, v_tang1_dt, v_tang2_dt = self.extension.radial_tangential_stretch(dx_tang, dy_tang, kwargs_lens,
                                                                            diff=self._diff)
        d_tang1 = v_tang1_dt - v_tang1
        d_tang2 = v_tang2_dt - v_tang2
        delta = np.sqrt(d_tang1**2 + d_tang2**2)
        if delta > 1:
            d_tang1 = v_tang1_dt + v_tang1
            d_tang2 = v_tang2_dt + v_tang2
            delta = np.sqrt(d_tang1 ** 2 + d_tang2 ** 2)
        curvature = delta / self._diff
        direction = np.arctan2(v_rad2 * np.sign(v_rad1 * self._x + v_rad2 * self._y), v_rad1 * np.sign(v_rad1 * self._x + v_rad2 * self._y))

        kwargs_arc = {'radial_stretch': radial_stretch,
                      'tangential_stretch': tangential_stretch,
                      'curvature': curvature,
                      'direction': direction,
                      'center_x': self._x, 'center_y': self._y}

        if self._tan_diff:
            raise Exception('not yet implemented')

        return kwargs_arc

    def hessian_fast(self, args):

        beta_x, beta_y = self._model_ra_pp.ray_shooting_fast(args)
        alpha_ra_pp, alpha_dec_pp = (self._x + self._diff/2) - beta_x, (self._y + self._diff/2) - beta_y

        beta_x, beta_y = self._model_ra_pn.ray_shooting_fast(args)
        alpha_ra_pn, alpha_dec_pn = (self._x + self._diff / 2) - beta_x, (self._y - self._diff / 2) - beta_y

        beta_x, beta_y = self._model_ra_np.ray_shooting_fast(args)
        alpha_ra_np, alpha_dec_np = (self._x - self._diff / 2) - beta_x, (self._y + self._diff / 2) - beta_y

        beta_x, beta_y = self._model_ra_nn.ray_shooting_fast(args)
        alpha_ra_nn, alpha_dec_nn = (self._x - self._diff / 2) - beta_x, (self._y - self._diff / 2) - beta_y

        f_xx = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / self._diff / 2
        f_xy = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / self._diff / 2
        f_yx = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / self._diff / 2
        f_yy = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / self._diff / 2

        return f_xx, f_xy, f_yx, f_yy

    def hessian_eigenvectors_fast(self, args):

        f_xx, f_xy, f_yx, f_yy = self.hessian_fast(args)
        A = np.array([[1 - f_xx, f_xy], [f_yx, 1 - f_yy]])
        w, v = np.linalg.eig(A)
        v11, v12, v21, v22 = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
        w1, w2 = w[0], w[1]
        return w1, w2, v11, v12, v21, v22

    def radial_tangential_stretch_fast(self, args, ra_0=0.0, dec_0=0.0, coordinate_frame_definitions=False):
        """

        """
        w1, w2, v11, v12, v21, v22 = self.hessian_eigenvectors_fast(args)
        v_x, v_y = self._x - ra_0, self._y - dec_0
        prod_v1 = v_x * v11 + v_y * v12
        prod_v2 = v_x * v21 + v_y * v22
        if (coordinate_frame_definitions is True and abs(prod_v1) >= abs(prod_v2)) or (
            coordinate_frame_definitions is False and w1 >= w2):
            lambda_rad = 1. / w1
            lambda_tan = 1. / w2
            v1_rad, v2_rad = v11, v12
            v1_tan, v2_tan = v21, v22
            prod_r = prod_v1
        else:
            lambda_rad = 1. / w2
            lambda_tan = 1. / w1
            v1_rad, v2_rad = v21, v22
            v1_tan, v2_tan = v11, v12
            prod_r = prod_v2
        if prod_r < 0:  # if radial eigenvector points towards the center
            v1_rad, v2_rad = -v1_rad, -v2_rad
        if v1_rad * v2_tan - v2_rad * v1_tan < 0:  # cross product defines orientation of the tangential eigenvector
            v1_tan *= -1
            v2_tan *= -1

        return lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan

    def radial_tangential_differentials_fast(self, args):

        kwargs_lens = self._param_class.args_to_kwargs(args)

        lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan = self.radial_tangential_stretch_fast(args)
        x0 = self._x
        y0 = self._y

        # computing angle of tangential vector in regard to the defined coordinate center
        cos_angle = (v1_tan * x0 + v2_tan * y0) / np.sqrt(
            (x0 ** 2 + y0 ** 2) * (v1_tan ** 2 + v2_tan ** 2))  # * np.sign(v1_tan * y0 - v2_tan * x0)
        orientation_angle = np.arccos(cos_angle) - np.pi / 2

        # computing differentials in tangential and radial directions
        dx_tan = self._x + self._diff * v1_tan
        dy_tan = self._y + self._diff * v2_tan
        lambda_rad_dtan, lambda_tan_dtan, v1_rad_dtan, v2_rad_dtan, v1_tan_dtan, v2_tan_dtan = self.extension.radial_tangential_stretch(
            dx_tan, dy_tan, kwargs_lens, diff=self._diff,
            ra_0=0.0, dec_0=0.0, coordinate_frame_definitions=True)
        dx_rad = self._x + self._diff * v1_rad
        dy_rad = self._y + self._diff * v2_rad
        lambda_rad_drad, lambda_tan_drad, v1_rad_drad, v2_rad_drad, v1_tan_drad, v2_tan_drad = self.extension.radial_tangential_stretch(
            dx_rad, dy_rad, kwargs_lens, diff=self._diff, ra_0=0.0, dec_0=0.0,
            coordinate_frame_definitions=True)

        # eigenvalue differentials in tangential and radial direction
        dlambda_tan_dtan = (lambda_tan_dtan - lambda_tan) / self._diff  # * np.sign(v1_tan * y0 - v2_tan * x0)
        dlambda_tan_drad = (lambda_tan_drad - lambda_tan) / self._diff  # * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_drad = (lambda_rad_drad - lambda_rad) / self._diff  # * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_dtan = (lambda_rad_dtan - lambda_rad) / self._diff  # * np.sign(v1_rad * x0 + v2_rad * y0)

        # eigenvector direction differentials in tangential and radial direction
        cos_dphi_tan_dtan = v1_tan * v1_tan_dtan + v2_tan * v2_tan_dtan  # / (np.sqrt(v1_tan**2 + v2_tan**2) * np.sqrt(v1_tan_dtan**2 + v2_tan_dtan**2))
        norm = np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_dtan ** 2 + v2_tan_dtan ** 2)
        cos_dphi_tan_dtan /= norm
        arc_cos_dphi_tan_dtan = np.arccos(np.abs(np.minimum(cos_dphi_tan_dtan, 1)))
        dphi_tan_dtan = arc_cos_dphi_tan_dtan / self._diff

        cos_dphi_tan_drad = v1_tan * v1_tan_drad + v2_tan * v2_tan_drad  # / (np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2))
        norm = np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2)
        cos_dphi_tan_drad /= norm
        arc_cos_dphi_tan_drad = np.arccos(np.abs(np.minimum(cos_dphi_tan_drad, 1)))
        dphi_tan_drad = arc_cos_dphi_tan_drad / self._diff

        cos_dphi_rad_drad = v1_rad * v1_rad_drad + v2_rad * v2_rad_drad  # / (np.sqrt(v1_rad**2 + v2_rad**2) * np.sqrt(v1_rad_drad**2 + v2_rad_drad**2))
        norm = np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_drad ** 2 + v2_rad_drad ** 2)
        cos_dphi_rad_drad /= norm
        cos_dphi_rad_drad = np.minimum(cos_dphi_rad_drad, 1)
        dphi_rad_drad = np.arccos(cos_dphi_rad_drad) / self._diff

        cos_dphi_rad_dtan = v1_rad * v1_rad_dtan + v2_rad * v2_rad_dtan  # / (np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2))
        norm = np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2)
        cos_dphi_rad_dtan /= norm
        cos_dphi_rad_dtan = np.minimum(cos_dphi_rad_dtan, 1)
        dphi_rad_dtan = np.arccos(cos_dphi_rad_dtan) / self._diff

        return lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan
