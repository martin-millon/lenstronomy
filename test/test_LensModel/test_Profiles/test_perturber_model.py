from lenstronomy.LensModel.Profiles.perturber_model import PerturberModel
from lenstronomy.LensModel.Profiles.sis import SIS
import numpy.testing as npt


class TestPerturberModel:

    def setup_method(self):
        self._sis = SIS()
        self.ra0 = 10
        self.dec0 = 5
        self._model = PerturberModel(profile=self._sis, ra_0=self.ra0, dec_0=self.dec0)

    def test_offset(self):

        # test that evaluations of deflections and hessian at ra0/dec0 is zero
        kwargs_lens = {"theta_E": 1, "center_x": 0, "center_y": 0}
        alpha_x, alpha_y = self._model.derivatives(self.ra0, self.dec0, **kwargs_lens)
        npt.assert_almost_equal(alpha_x, 0, decimal=10)
        npt.assert_almost_equal(alpha_y, 0, decimal=10)

        f_xx, f_xy, f_yx, f_yy = self._model.hessian(self.ra0, self.dec0, **kwargs_lens)
        npt.assert_almost_equal(f_xx, 0, decimal=10)
        npt.assert_almost_equal(f_xy, 0, decimal=10)
        npt.assert_almost_equal(f_yx, 0, decimal=10)
        npt.assert_almost_equal(f_yy, 0, decimal=10)

        f = self._model.function(self.ra0, self.dec0, **kwargs_lens)
        f_sis = self._sis.function(self.ra0, self.dec0, **kwargs_lens)
        npt.assert_almost_equal(f, 0, decimal=10)
