"""Run with python -m unittest discover -s tests -v (NumPy/SciPy/Plotly required).

Execute the notebook's actual definitions, not a copied production kernel.
The independent oracle lives in the self-contained notebook for Colab users.
"""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

NOTEBOOK = Path(__file__).resolve().parents[1] / 'tutorials/04_Polychromatic_Electric_Field_Knots.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']


def cell_starting(prefix):
    return next(''.join(c['source']) for c in CELLS
                if c['cell_type'] == 'code' and ''.join(c['source']).startswith(prefix))


def namespace(preset='paper_132', z=0., lambda_ref=1.):
    ns = {}
    imports = cell_starting('%pip')
    exec('\n'.join(line for line in imports.splitlines() if not line.startswith('%')), ns)
    settings = cell_starting('# --- User controls ---')
    settings = settings.replace('PRESET = "paper_132"', f'PRESET = "{preset}"')
    settings = settings.replace('Z = 0.0 ', f'Z = {z} ')
    settings = settings.replace('LAMBDA_REF, F = 1.0, 1.0', f'LAMBDA_REF, F = {lambda_ref}, 2.4')
    settings = settings.replace('HALF_WIDTH, NXY = 1.0, 81', 'HALF_WIDTH, NXY = 1.0, 21')
    with contextlib.redirect_stdout(io.StringIO()):
        exec(settings, ns)
        exec(cell_starting('R_JONES='), ns)
        exec(cell_starting('def direct_rw_basis'), ns)
    return ns


class PhysicsTests(unittest.TestCase):
    def test_complex_fields_both_presets_and_axial_planes(self):
        for preset in ('paper_132', 'paper_235'):
            for z in (0., .31):
                with self.subTest(preset=preset, z=z):
                    ns = namespace(preset, z, lambda_ref=1.7)
                    worst = max(max(r[c] for c in ('ex', 'ey', 'ez'))
                                for r in ns['verification'])
                    self.assertLess(worst, 5e-4)
                    # Check superposed modes too, preserving their intrinsic phases.
                    point = [[ns['x'][12], ns['y'][8], z]]
                    for j, ((ell, delta), h, weight) in enumerate(zip(
                            ns['modes'], ns['harmonics'], ns['weights'])):
                        bp = ns['direct_rw_basis'](ell, h, point)[0]
                        bm = ns['direct_rw_basis'](-ell, h, point)[0]
                        direct = weight * (ns['R_JONES'] @ bp + np.exp(1j*delta)*
                                           (ns['L_JONES'] @ bm))/np.sqrt(2)
                        actual = ns['E_colors'][j, :, 8, 12]
                        np.testing.assert_allclose(actual, direct, atol=2e-4, rtol=5e-4)

    def test_radial_on_axis_and_azimuthal_longitudinal_symmetry(self):
        ns = namespace()
        # Radial input: finite on-axis longitudinal field, zero transverse field.
        radial = ns['vector_vortex'](1, 0, 1)
        self.assertLess(np.linalg.norm(radial[:2, 10, 10]), 1e-12)
        self.assertGreater(abs(radial[2, 10, 10]), 1e-2)
        azimuthal = ns['vector_vortex'](1, np.pi, 1)
        self.assertLess(np.max(abs(azimuthal[2])), 1e-12)

    def test_temporal_reconstruction_matches_figure1(self):
        tau = np.linspace(0, 2*np.pi, 1601)
        # Re[(a+ib)e^-imt] = a cos(mt)+b sin(mt).
        phasors = np.array([[0, 1, 1j], [0, -2, 2j], [1j, 0, 0]])
        curve = np.real(np.einsum('mc,mt->tc', phasors,
                                  np.exp(-1j*np.outer([1, 2, 3], tau))))
        analytic = np.column_stack((np.sin(3*tau), np.cos(tau)-2*np.cos(2*tau),
                                    np.sin(tau)+2*np.sin(2*tau)))
        np.testing.assert_allclose(curve, analytic, atol=1e-14)
        np.testing.assert_allclose(curve[0], curve[-1], atol=1e-14)
        reverse = np.real(np.einsum('mc,mt->tc', phasors,
                                    np.exp(1j*np.outer([1, 2, 3], tau))))
        np.testing.assert_allclose(reverse, curve[::-1], atol=2e-14)

    def test_invalid_harmonics_are_rejected(self):
        for harmonics in ((1, 1.5, 2.5), (1, 1, 2), (0, 2, 3), (1, 2, np.nan)):
            ns = dict(np=np)
            settings = cell_starting('# --- User controls ---')
            # Only execute validation block; no focusing needed for rejected data.
            settings = settings[settings.index('presets ='):settings.index('assert len(harmonics)')]
            ns.update(PRESET='custom', CUSTOM_HARMONICS=harmonics)
            with self.subTest(harmonics=harmonics), self.assertRaises(ValueError):
                exec(settings, ns)


if __name__ == '__main__':
    unittest.main()
