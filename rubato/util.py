class MultiplaneFast(object):

    """
    This class accelerates ray tracing computations in multi plane lensing for quadruple image lenses by only
    computing the deflection from objects in front of the main deflector at z_lens one time. The first ray tracing
    computation through the foreground is saved and re-used, but it will always have the same shape as the initial
    x_image, y_image arrays.

    """

    def __init__(self, x_image, y_image, lensModel, lens_model_to_vary, lens_model_fixed,
                 param_class, foreground_rays=None, tol_source=1e-5):

        """
        This creates the class from a specified set of lens models that have already been created, thus saving
        memory if several instances of MultiplaneFast need to be created for a fixed lens model.

        :param x_image: x_image to fit
        :param y_image: y_image to fit
        :param lensModel: an instance of LensModel that contains every deflector in the lens system
        :param lens_model_to_vary: an instance of LensModel that contains only the lens models whose keywords are
        being sampled in the optimization
        :param lens_model_fixed: an instance of LensModel that contains the lens models whose properties are being
        held fixed during the optimization
        :param param_class: an instance of ParamClass (see documentation in QuadOptimmizer.param_manager)
        :param foreground_rays: (optional) pre-computed foreground rays from a previous iteration, if they are not specified
        they will be re-computed
        :param tol_source: source plane chi^2 sigma
        :param numerical_alpha_class: class for computing numerically tabulated deflection angles
        """

        self.lens_model_to_vary = lens_model_to_vary
        self.lensModel = lensModel
        self.lens_model_fixed = lens_model_fixed

        self._z_lens = lensModel.z_lens
        self._z_source = lensModel.z_source
        self._x_image = x_image
        self._y_image = y_image
        self._param_class = param_class
        self._tol_source = tol_source
        self._foreground_rays = foreground_rays

    @classmethod
    def fromModelList(cls, x_image, y_image, z_lens, z_source, lens_model_list, redshift_list,
                      astropy_instance, param_class, foreground_rays=None,
                      tol_source=1e-5, numerical_alpha_class=None):

        """
        This creates the class from a list of lens models and redshifts. The lens model list and redshift list
        will be split at the value of "to_vary_index" specified in the param_class (see classes in param_manager).
        Since this method creates several lens model classes it can consume significant memory.

        :param x_image: x_image to fit
        :param y_image: y_image to fit
        :param z_lens: lens redshift
        :param z_source: source redshift
        :param lens_model_list: list of lens models
        :param redshift_list: list of lens redshifts
        :param astropy_instance: instance of astropy to pass to lens model
        :param param_class: an instance of ParamClass (see documentation in QuadOptimmizer.param_manager)
        :param foreground_rays: (optional) pre-computed foreground rays from a previous iteration, if they are not specified
        they will be re-computed
        :param tol_source: source plane chi^2 sigma
        :param numerical_alpha_class: class for computing numerically tabulated deflection angles
        :return:
        """
        lensModel = LensModel(lens_model_list, z_lens, z_source, redshift_list, astropy_instance,
                              multi_plane=True, numerical_alpha_class=numerical_alpha_class)
        lensmodel_list_to_vary = lens_model_list[0:param_class.to_vary_index]
        redshift_list_to_vary = redshift_list[0:param_class.to_vary_index]
        lensmodel_list_fixed = lens_model_list[param_class.to_vary_index:]
        redshift_list_fixed = redshift_list[param_class.to_vary_index:]
        lens_model_to_vary = LensModel(lensmodel_list_to_vary, z_lens, z_source, redshift_list_to_vary,
                                       cosmo=astropy_instance, multi_plane=True,
                                       numerical_alpha_class=numerical_alpha_class)
        lens_model_fixed = LensModel(lensmodel_list_fixed, z_lens, z_source, redshift_list_fixed,
                                     cosmo=astropy_instance, multi_plane=True,
                                     numerical_alpha_class=numerical_alpha_class)

        return MultiplaneFast(x_image, y_image, lensModel, lens_model_to_vary, lens_model_fixed,
                              param_class, foreground_rays, tol_source)


    def chi_square(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: total chi^2 penalty (source chi^2 + param chi^2), where param chi^2 is computed by the specified
         param_class
        """
        source_plane_penlty = self.source_plane_chi_square(args_lens)

        param_penalty = self._param_class.param_chi_square_penalty(args_lens)

        return source_plane_penlty + param_penalty

    def logL(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: the log likelihood corresponding to the given chi^2
        """
        chi_square = self.chi_square(args_lens)

        return -0.5 * chi_square

    def source_plane_chi_square(self, args_lens, *args, **kwargs):

        """

        :param args_lens: array of lens model parameters being optimized, computed from kwargs_lens in a specified
         param_class, see documentation in QuadOptimizer.param_manager
        :return: chi2 penalty for the source position (all images must map to the same source coordinate)
        """

        betax, betay = self.ray_shooting_fast(args_lens)

        dx_source = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (
                betax[0] - betax[3]) ** 2 + (
                             betax[1] - betax[2]) ** 2 +
                     (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy_source = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (
                betay[0] - betay[3]) ** 2 + (
                             betay[1] - betay[2]) ** 2 +
                     (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        chi_square = 0.5 * (dx_source + dy_source) / self._tol_source ** 2

        return chi_square

    def alpha_fast(self, args_lens):
        """
        Performs a ray tracing computation through observed coordinates on the sky (self._x_image, self._y_image)
        to the source plane coordinate beta. Returns the deflection angle alpha:

        beta = x - alpha(x)

        :param args_lens: An array of parameters being optimized. The array is computed from a set of key word arguments
        by an instance of ParamClass (see documentation in QuadOptimizer.param_manager)
        :return: the xy coordinate of each ray traced back to the source plane
        """

        betax, betay = self.ray_shooting_fast(args_lens)
        alpha_x = self._x_image - betax
        alpha_y = self._y_image - betay

        return alpha_x, alpha_y

    def ray_shooting_fast(self, args_lens):

        """
        Performs a ray tracing computation through observed coordinates on the sky (self._x_image, self._y_image)
        to the source plane, returning the final coordinates of each ray on the source plane

        :param args_lens: An array of parameters being optimized. The array is computed from a set of key word arguments
         by an instance of ParamClass (see documentation in QuadOptimizer.param_manager)
        :return: the xy coordinate of each ray traced back to the source plane
        """

        # these do not depend on kwargs_lens_array
        x, y, alpha_x, alpha_y = self._ray_shooting_fast_foreground()

        # convert array into new kwargs dictionary
        kw = self._param_class.args_to_kwargs(args_lens)
        index = self._param_class.to_vary_index
        kwargs_lens = kw[0:index]
        # evaluate main deflector deflection angles
        x, y, alpha_x, alpha_y = self.lens_model_to_vary.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self._z_lens, self._z_lens, kwargs_lens, include_z_start=True)

        # ray trace through background halos
        kwargs_lens = kw[index:]
        x, y, _, _ = self.lens_model_fixed.lens_model.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self._z_lens, self._z_source, kwargs_lens, check_convention=False)

        beta_x, beta_y = self.lens_model_fixed.lens_model.co_moving2angle_source(x, y)

        return beta_x, beta_y

    def _ray_shooting_fast_foreground(self):

        """
        Does the ray tracing through the foreground halos only once
        """

        if self._foreground_rays is None:

            # These do not depend on the kwargs being optimized for
            kw = self._param_class.kwargs_lens
            index = self._param_class.to_vary_index
            kwargs_lens = kw[index:]

            x0, y0 = np.zeros_like(self._x_image), np.zeros_like(self._y_image)
            x, y, alpha_x, alpha_y = self.lens_model_fixed.lens_model.ray_shooting_partial(
                x0, y0, self._x_image, self._y_image, z_start=0.,
                                                         z_stop=self._z_lens, kwargs_lens=kwargs_lens)

            self._foreground_rays = (x, y, alpha_x, alpha_y)

        return self._foreground_rays[0], self._foreground_rays[1], self._foreground_rays[2], self._foreground_rays[3]



class MultiplaneFastDifferential(object):
    """
    This class uses the ray tracing routines in MultiPlaneFast to compute numerical derivatives of deflection angles
    i.e. the components of the hessian matrix
    """

    def __init__(self, diff, x, y, img_plane_x, img_plane_y, z_lens, z_source, lens_model_list, redshift_list,
                 astropy_instance, param_class, numerical_alpha_class=None):

        """

        :param diff: the angular scale over which to compute the finite difference derivative
        :param x: the x coordinate where the derivatives are taken
        :param y: the y coordinate where the derivatives are taken
        :param img_plane_x: the x-angular coordinate where a ray hits the plane lens plane
        :param img_plane_y: the y-angular coordinate where a ray hits the plane lens plane
        :param z_lens: the lens redshift
        :param z_source: the source redshift
        :param lens_model_list: the list of lens models to be passed to MultiPlaneFast
        :param redshift_list: the list of deflector redshifts to be passed to MuliPlaneFast
        :param astropy_instance: an instance as astropy
        :param param_class: the param class that defintes the function being minimized (see param_manager)
        :param numerical_alpha_class: (optional) a class that returns deflection angles for a numerically
        integrated mass profile
        """

        # observed (x,y)
        self._x = x
        self._y = y

        # lensed (x,y), where ray hits main lens plane
        self._img_plane_x = img_plane_x
        self._img_plane_y = img_plane_y

        self._diff = diff
        self._param_class = param_class
        self._fast_ray_shooting_dx_plus = MultiplaneFast.fromModelList(x + diff / 2, y, z_lens, z_source,
                                                                       lens_model_list, redshift_list, astropy_instance,
                                                                       param_class, None,
                                                                       numerical_alpha_class=numerical_alpha_class)

        lens_model_to_vary = self._fast_ray_shooting_dx_plus.lens_model_to_vary
        lens_model = self._fast_ray_shooting_dx_plus.lensModel
        lens_model_fixed = self._fast_ray_shooting_dx_plus.lens_model_fixed

        self._fast_ray_shooting_dy_plus = MultiplaneFast(x, y + diff / 2, lens_model, lens_model_to_vary,
                                                         lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus = MultiplaneFast(x - diff / 2, y, lens_model, lens_model_to_vary,
                                                          lens_model_fixed, param_class)

        self._fast_ray_shooting_dy_minus = MultiplaneFast(x, y - diff / 2, lens_model, lens_model_to_vary,
                                                          lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_plus_dy_plus = MultiplaneFast(x + diff / 2, y + diff / 2,
                                                                 lens_model, lens_model_to_vary,
                                                                 lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_plus_dy_minus = MultiplaneFast(x + diff / 2, y - diff / 2, lens_model,
                                                                  lens_model_to_vary,
                                                                  lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus_dy_minus = MultiplaneFast(x - diff / 2, y - diff / 2, lens_model,
                                                                   lens_model_to_vary,
                                                                   lens_model_fixed, param_class)

        self._fast_ray_shooting_dx_minus_dy_plus = MultiplaneFast(x - diff / 2, y + diff / 2, lens_model,
                                                                  lens_model_to_vary,
                                                                  lens_model_fixed, param_class)

    def hessian(self, args, diff_method='square'):

        """

        :param args: the array of lens model args being optimized (see param_manager)
        :param diff_method: the method for calculating the derivatives, options include cross, square, and
        average, where average is the mean of cross and square
        :return: the derivatives of the deflection angles computed using the specified diff_methdd
        """

        if diff_method == 'cross':
            f_xx, f_xy, f_yx, f_yy = self._hessian_cross(args)
        elif diff_method == 'square':
            f_xx, f_xy, f_yx, f_yy = self._hessian_square(args)
        elif diff_method == 'average':
            _fxx, _fxy, _fyx, _fyy = self._hessian_cross(args)
            fxx_, fxy_, fyx_, fyy_ = self._hessian_square(args)
            f_xx = 0.5 * (fxx_ + _fxx)
            f_xy = 0.5 * (fxy_ + _fxy)
            f_yx = 0.5 * (fyx_ + _fyx)
            f_yy = 0.5 * (fyy_ + _fyy)
        else:
            raise Exception('diff method must be either cross, square, or average')

        return f_xx, f_xy, f_yx, f_yy

    def _hessian_eigenvectors_fast(self, args):
        """
        computes magnification eigenvectors at the given position

        :param args: the array of lens model args being optimized (see param_manager)
        :param diff_method: the method for calculating the derivatives, options include cross, square, and
        average, where average is the mean of cross and square
        :return: radial stretch, tangential stretch
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(args)
        A = np.array([[1 - f_xx, f_xy], [f_yx, 1 - f_yy]])
        w, v = np.linalg.eig(A)
        v11, v12, v21, v22 = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
        w1, w2 = w[0], w[1]
        return w1, w2, v11, v12, v21, v22

    def _radial_tangential_stretch_fast(self, args, coordinate_frame_definitions=False):
        """
        computes the radial and tangential stretches at a given position

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :param diff: float or None, finite average differential scale
        :return: radial stretch, tangential stretch
        """

        ra_0 = self._img_plane_y
        dec_0 = self._img_plane_x
        x = self._x
        y = self._y

        w1, w2, v11, v12, v21, v22 = self._hessian_eigenvectors_fast(args)
        v_x, v_y = x - ra_0, y - dec_0

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
        radial_stretch, tangential_stretch, v_rad1, v_rad2, v_tang1, v_tang2 = self._radial_tangential_stretch_fast(
            args)

        _, _, _, _, v_tang1_dt, v_tang2_dt = self._radial_tangential_stretch_fast(args)
        d_tang1 = v_tang1_dt - v_tang1
        d_tang2 = v_tang2_dt - v_tang2
        delta = np.sqrt(d_tang1 ** 2 + d_tang2 ** 2)
        if delta > 1:
            d_tang1 = v_tang1_dt + v_tang1
            d_tang2 = v_tang2_dt + v_tang2
            delta = np.sqrt(d_tang1 ** 2 + d_tang2 ** 2)
        curvature = delta / self._diff
        direction = np.arctan2(v_rad2 * np.sign(v_rad1 * self._x + v_rad2 * self._y),
                               v_rad1 * np.sign(v_rad1 * self._x + v_rad2 * self._y))

        return (radial_stretch, tangential_stretch, curvature, direction)

    def _hessian_cross(self, args):
        """

        :param args: the array of lens model args being optimized (see param_manager)
        :return: the derivatives of the deflection angles
        """

        alpha_ra_dx, alpha_dec_dx = self._fast_ray_shooting_dx_plus.alpha_fast(args)
        alpha_ra_dy, alpha_dec_dy = self._fast_ray_shooting_dy_plus.alpha_fast(args)

        alpha_ra_dx_, alpha_dec_dx_ = self._fast_ray_shooting_dx_minus.alpha_fast(args)
        alpha_ra_dy_, alpha_dec_dy_ = self._fast_ray_shooting_dy_minus.alpha_fast(args)

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_) / self._diff / 2
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_) / self._diff / 2
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_) / self._diff / 2
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_) / self._diff / 2

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def _hessian_square(self, args):
        """

        :param args: the array of lens model args being optimized (see param_manager)
        :return: the derivatives of the deflection angles
        """

        alpha_ra_pp, alpha_dec_pp = self._fast_ray_shooting_dx_plus_dy_plus.alpha_fast(args)
        alpha_ra_pn, alpha_dec_pn = self._fast_ray_shooting_dx_plus_dy_minus.alpha_fast(args)

        alpha_ra_np, alpha_dec_np = self._fast_ray_shooting_dx_minus_dy_plus.alpha_fast(args)
        alpha_ra_nn, alpha_dec_nn = self._fast_ray_shooting_dx_minus_dy_minus.alpha_fast(args)

        f_xx = (alpha_ra_pp - alpha_ra_np + alpha_ra_pn - alpha_ra_nn) / self._diff / 2
        f_xy = (alpha_ra_pp - alpha_ra_pn + alpha_ra_np - alpha_ra_nn) / self._diff / 2
        f_yx = (alpha_dec_pp - alpha_dec_np + alpha_dec_pn - alpha_dec_nn) / self._diff / 2
        f_yy = (alpha_dec_pp - alpha_dec_pn + alpha_dec_np - alpha_dec_nn) / self._diff / 2

        return f_xx, f_xy, f_yx, f_yy
