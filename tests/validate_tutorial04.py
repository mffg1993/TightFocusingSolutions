"""Execute every code cell sequentially in a headless IPython session.

Run: python tests/validate_tutorial04.py
This exercises Python, rich-output construction, widgets and NPZ export, but does
not test a Colab browser frontend or the optional pyknotid installation.
"""
import json
import os
from pathlib import Path
import tempfile
import time

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import plotly.io as pio

notebook = Path(__file__).resolve().parents[1] / 'tutorials/04_Polychromatic_Electric_Field_Knots.ipynb'
cells = json.loads(notebook.read_text())['cells']
shell = InteractiveShell.instance()
pio.renderers.default = 'json'
start = time.monotonic()
with tempfile.TemporaryDirectory(prefix='tutorial04-') as temp:
    previous = Path.cwd()
    try:
        os.chdir(temp)
        for i, cell in enumerate(cells):
            if cell['cell_type'] != 'code':
                continue
            with capture_output() as output:
                result = shell.run_cell(''.join(cell['source']))
            print(f'cell {i}: {"PASS" if result.success else "FAIL"}', flush=True)
            if not result.success:
                print(output.stdout, output.stderr)
                result.raise_error()
            # Print scientific diagnostics, avoiding large rendered figure payloads.
            if ''.join(cell['source']).startswith(('def direct_rw_basis', 'def focus_convergence')):
                print(output.stdout[:12000])
        import numpy as np
        with np.load('polychromatic_electric_knots_fields.npz', allow_pickle=False) as saved:
            assert saved['E_colors'].shape == (3, 3, 81, 81)
            assert np.isfinite(saved['E_colors']).all()
            assert str(saved['schema_version']) == 'tutorial04-validation-v1'
    finally:
        os.chdir(previous)
print(f'All code cells and NPZ reload passed in {time.monotonic()-start:.1f}s')
