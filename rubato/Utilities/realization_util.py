from rubato.Utilities.ray_tracing_util import interpolate_ray_paths
from scipy.interpolate import interp1d
import numpy as np

def setup_realization(realization, model_init_list,
                        aperture_radius=0.3,
                        log_min_mass_aperture=0.0,
                        log_min_mass_global=7.0):

    ray_interp_x = []
    ray_interp_y = []

    for model_init in model_init_list:

        kwargs_init = model_init.kwargs_init
        source_x = kwargs_init['source_x']
        source_y = kwargs_init['source_y']
        z_source = kwargs_init['z_source']
        x_image, y_image = kwargs_init['x_image'], kwargs_init['y_image']
        kwargs_lens_macro = kwargs_init['kwargs_lens_other']
        lens_model_shift = model_init.lensmodel_shift()
        kwargs_shift = model_init.compute_kwargs_shift()
        kwargs_init = [kwargs_shift] + kwargs_lens_macro
        _ray_interp_x, _ray_interp_y, distances = interpolate_ray_paths([x_image], [y_image], lens_model_shift, kwargs_init, z_source,
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

    realization_final_list = []
    lens_model_list_halos = []
    redshift_array_halos = []
    kwargs_halos = []

    for i, model_init in enumerate(model_init_list):

        realization_final = realization_shifted.filter(
            aperture_radius,
            aperture_radius,
            log_min_mass_aperture,
            log_min_mass_aperture,
            log_min_mass_global,
            log_min_mass_global,
            [ray_interp_x[i]], [ray_interp_y[i]]
        )
        realization_final_list.append(realization_final)
        lm_list, zarray, kw, _ = realization_final.lensing_quantities()
        lens_model_list_halos.append(lm_list)
        redshift_array_halos.append(list(zarray))
        kwargs_halos.append(kw)

    return lens_model_list_halos, list(redshift_array_halos), kwargs_halos, realization_final_list
