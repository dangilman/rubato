import numpy as np


class CurvedArcBase(object):

    def __init__(self, kwargs_lens_init):

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    @property
    def to_vary_index(self):
        """
        The number of lens models being varied in this routine. This is set to 1 because only the first lens model
        is being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 1

class CurvedArcTanDiff(CurvedArcBase):

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        radial_stretch = kwargs[0]['radial_stretch']
        tangential_stretch = kwargs[0]['tangential_stretch']
        curvature = kwargs[0]['curvature']
        direction = kwargs[0]['direction']
        dtan_dtan = kwargs[0]['dtan_dtan']
        args = (radial_stretch, tangential_stretch, curvature, direction, dtan_dtan)

        return args

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            d_radial_stretch = 2.
            d_tangential_stretch = 2.
            d_direction = np.pi / 10
            d_curvature = 2.
            d_dtan_dtan = 1.5

        else:
            d_radial_stretch = 5.0
            d_tangential_stretch = 5.0
            d_direction = np.pi / 2
            d_curvature = 5.
            d_dtan_dtan = 4.0

        shifts = np.array([d_radial_stretch, d_tangential_stretch, d_curvature, d_direction, d_dtan_dtan]) * scale

        low, high = np.empty_like(shifts), np.empty_like(shifts)

        for i in range(0, len(shifts)):
            low[i] = args[i] / shifts[i]
            high[i] = args[i] * shifts[i]

        return low, high

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (radial_stretch, tangential_stretch, curvature, direction, dtan_dtan) = args

        center_x = self.kwargs_lens[0]['center_x']
        center_y = self.kwargs_lens[0]['center_y']

        kwargs = {'radial_stretch': radial_stretch, 'tangential_stretch': tangential_stretch,
                  'curvature': curvature, 'direction': direction, 'dtan_dtan': dtan_dtan,
                  'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs

        return self.kwargs_lens

class CurvedArc(CurvedArcBase):

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        radial_stretch = kwargs[0]['radial_stretch']
        tangential_stretch = kwargs[0]['tangential_stretch']
        curvature = kwargs[0]['curvature']
        direction = kwargs[0]['direction']

        args = (radial_stretch, tangential_stretch, curvature, direction)

        return args

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            d_radial_stretch = 2.
            d_tangential_stretch = 2.
            d_direction = np.pi / 10
            d_curvature = 2.

        else:
            d_radial_stretch = 10
            d_tangential_stretch = 10
            d_direction = np.pi / 2
            d_curvature = 10.

        shifts = np.array([d_radial_stretch, d_tangential_stretch, d_curvature, d_direction]) * scale

        low, high = np.empty_like(shifts), np.empty_like(shifts)

        for i in range(0, len(shifts)):
            low[i] = args[i] / shifts[i]
            high[i] = args[i] * shifts[i]

        return low, high

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (radial_stretch, tangential_stretch, curvature, direction) = args

        center_x = self.kwargs_lens[0]['center_x']
        center_y = self.kwargs_lens[0]['center_y']

        kwargs = {'radial_stretch': radial_stretch, 'tangential_stretch': tangential_stretch,
         'curvature': curvature, 'direction': direction, 'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs

        return self.kwargs_lens

class ShiftLensModelParamManager(object):

    def __init__(self, kwargs_lens_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    @property
    def to_vary_index(self):
        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 1

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            dx = 0.1
            dy = 0.1

        else:
            dx = 0.5
            dy = 0.5

        shifts = np.array([dx, dy])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        alpha_x_shift, alpha_y_shift = kwargs[0]['alpha_x_shift'], kwargs[1]['alpha_y_shift']
        args = (alpha_x_shift, alpha_y_shift)
        return args

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (alpha_x_shift, alpha_y_shift) = args
        self.kwargs_lens[0] = {'alpha_x': alpha_x_shift, 'alpha_y': alpha_y_shift}
        return self.kwargs_lens

