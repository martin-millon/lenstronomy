__author__ = 'sibirrer'
from lenstronomy.GalKin.galkin_model import GalkinModel


import numpy.testing as npt
import numpy as np
import pytest
import unittest


class TestGalkinModel(object):

    def setup(self):
        pass

    def test_radius_slope_anisotropy(self):

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_model = {'anisotropy_model': 'OM', 'mass_profile_list': ['SPP'], 'light_profile_list': ['HERNQUIST']}
        kwargs_numerics = {'interpol_grid_num': 1000, 'log_integration': True,
                           'max_integrate': 100, 'min_integrate': 0.001}
        kin_analytic = GalkinModel(kwargs_model, kwargs_cosmo, analytic_kinematics=True, kwargs_numerics=kwargs_numerics)
        r = 1
        theta_E, gamma = 1, 2.
        a_ani = 10
        r_eff = 0.1
        out = kin_analytic.check_df(r, kwargs_mass={'theta_E': theta_E, 'gamma': gamma}, kwargs_light={'r_eff': r_eff},
                           kwargs_anisotropy={'r_ani': a_ani*r_eff})
        assert out > 0
        print(out)


        kin_numeric = GalkinModel(kwargs_model, kwargs_cosmo, analytic_kinematics=False, kwargs_numerics=kwargs_numerics)
        out_num = kin_numeric.check_df(r, kwargs_mass=[{'theta_E': theta_E, 'gamma': gamma}],
                                       kwargs_light=[{'Rs': r_eff * 0.551, 'amp': 1}], kwargs_anisotropy={'r_ani': a_ani*r_eff})
        assert out_num > 1
        npt.assert_almost_equal(out_num/out, 1, decimal=2)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # r_list = np.logspace(-1, 1, 20)
        # crit_list = []
        # for r in r_list:
        #    crit_list.append(kin.check_df(r, theta_E, gamma, a_ani, r_eff))
        # plt.plot(r_list, crit_list)
        # plt.show()
        # assert 1 == 0


if __name__ == '__main__':
    pytest.main()