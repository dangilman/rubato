import pytest
import numpy.testing as npt
from rubato.Utilities.lens_model_util import MultiplaneFastDifferential
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from time import time
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFreeShear

class TestFastRayTracing(object):

    def setup(self):

        self.x_image, self.y_image = -1.148, 0.381
        self.zlens, self.zsource = 0.5, 1.5
        self.lens_model_list = ['EPL', 'SHEAR', 'SIS', 'SIS', 'SIS', 'SIS', 'SIS', 'SIS']
        self.redshift_list = [self.zlens, self.zlens, self.zlens-0.2, self.zlens, self.zlens + 0.1,
                              self.zlens - 0.2, self.zlens, self.zlens + 0.7]
        self.kwargs_lens = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1':-0.1, 'e2': 0.2, 'gamma': 2.05},
                       {'gamma1': 0.04, 'gamma2': 0.01}, {'theta_E': 0.05, 'center_x': 0.6, 'center_y': -0.5},
                            {'theta_E': 0.1, 'center_x': 0.9, 'center_y': 0.4},
                            {'theta_E': 0.05, 'center_x': -0.6, 'center_y': -0.5},
                            {'theta_E': 0.05, 'center_x': self.x_image - 0.01, 'center_y': self.y_image + 0.1},
                            {'theta_E': 0.025, 'center_x': self.x_image + 0.06, 'center_y': self.y_image - 0.02},
                            {'theta_E': 0.035, 'center_x': -0.2, 'center_y': 0.0}]
        self.lens_model = LensModel(self.lens_model_list, lens_redshift_list=self.redshift_list, multi_plane=True,
                                    z_source=self.zsource)
        self.extension = LensModelExtensions(self.lens_model)

        self.astropy = Background().cosmo
        self.param_class = PowerLawFreeShear(self.kwargs_lens)

    def test_hessian(self):

        smoothing = 0.001
        self.fast_multiplane = MultiplaneFastDifferential(self.x_image, self.y_image, self.zlens, self.zsource, self.lens_model,
                                                          self.lens_model_list, self.redshift_list,
                                                          self.astropy, smoothing, self.param_class)

        fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x_image, self.y_image, self.kwargs_lens, diff=smoothing)

        args = self.param_class.kwargs_to_args(self.kwargs_lens)
        fxx_fast, fxy_fast, fyx_fast, fyy_fast = self.fast_multiplane.hessian_fast(args)
        npt.assert_almost_equal(fxx, fxx_fast)
        npt.assert_almost_equal(fyy, fyy_fast)
        npt.assert_almost_equal(fxy, fxy_fast)
        npt.assert_almost_equal(fyx, fyx_fast)

    def test_hessian_eigenvectors(self):

        smoothing = 0.03
        self.fast_multiplane = MultiplaneFastDifferential(self.x_image, self.y_image, self.zlens, self.zsource, self.lens_model,
                                                          self.lens_model_list, self.redshift_list,
                                                          self.astropy, smoothing, self.param_class)

        w1, w2, v11, v12, v21, v22 = self.extension.hessian_eigenvectors(self.x_image, self.y_image,
                                                                         self.kwargs_lens, diff=smoothing)

        args = self.param_class.kwargs_to_args(self.kwargs_lens)
        w1_fast, w2_fast, v11_fast, v12_fast, v21_fast, v22_fast = self.fast_multiplane.hessian_eigenvectors_fast(args)
        npt.assert_almost_equal(w1, w1_fast)
        npt.assert_almost_equal(w2, w2_fast)
        npt.assert_almost_equal(v11, v11_fast)
        npt.assert_almost_equal(v12, v12_fast)
        npt.assert_almost_equal(v21, v21_fast)
        npt.assert_almost_equal(v22, v22_fast)

    def test_curved_arc_estimate(self):

        smoothing = 0.03
        self.fast_multiplane = MultiplaneFastDifferential(self.x_image, self.y_image, self.zlens, self.zsource, self.lens_model,
                                                          self.lens_model_list, self.redshift_list,
                                                          self.astropy, smoothing, self.param_class)
        kwargs_true = self.extension.curved_arc_estimate(self.x_image, self.y_image,
                                                        self.kwargs_lens, smoothing=smoothing, smoothing_3rd=smoothing)

        args = self.param_class.kwargs_to_args(self.kwargs_lens)
        kwargs_model = self.fast_multiplane.curved_arc_estimate_fast(args)
        for key in kwargs_true.keys():
            npt.assert_almost_equal(kwargs_model[key], kwargs_true[key])


if __name__ == '__main__':
    pytest.main()
