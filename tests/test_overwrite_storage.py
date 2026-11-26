import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_scientific_workflow import modal_fixture,write_saved_run
from test_spatial_workflow import classification_table
from linbo3.pipeline import run
from linbo3.storage import set_profile,analysis_root,remove_owned
from linbo3.spatial_workflow import prepare_spatial_study,run_spatial_study

class OverwriteStorage(unittest.TestCase):
 def tearDown(self):set_profile('main')
 def test_comsol_one_current_dataset_without_duplicate_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'model.mph').write_bytes(b'model')
   cfg=dict(model='model.mph',output_dir='results',widths_um=[.5,.7],geometry={},
    bands={'PUMP':dict(start_um=.7,stop_um=.85,points=9),'IS':dict(start_um=1.4,stop_um=1.7,points=9)},
    modes=None,cores=2,field_grid=[2,2],pump_wavelength_um=.775)
   path=root/'config.json'
   def solve(model,wavelengths,**kw):
    kw['log'].write_text('complete');return modal_fixture('PUMP' if wavelengths[0]<1 else 'IS')
   with patch('linbo3.pipeline.solve_modal_fields',side_effect=solve) as solver:
    path.write_text(json.dumps(cfg));first=run(path,stage='comsol');self.assertEqual(solver.call_count,4)
    self.assertEqual(run(path,stage='comsol'),first);self.assertEqual(solver.call_count,4)
    (first/'analysis').mkdir();(first/'analysis/stale').write_text('old')
    cfg['widths_um']=[.5,.7,.9];path.write_text(json.dumps(cfg))
    self.assertEqual(run(path,stage='comsol'),first);self.assertEqual(solver.call_count,6)
    self.assertFalse((first/'analysis').exists())
    self.assertEqual(len(list((root/'results').glob('*'))),1)
    self.assertEqual(len(list(first.rglob('*.npz'))),6)
    cfg['widths_um']=[.5];path.write_text(json.dumps(cfg));run(path,stage='comsol')
    self.assertEqual(len(list(first.glob('w_*'))),1)
 def test_python_new_grid_replaces_old_and_profiles_are_isolated(self):
  with tempfile.TemporaryDirectory() as tmp:
   raw=write_saved_run(Path(tmp)/'run');spatial=prepare_spatial_study(raw,classification_table())
   with patch('linbo3.spatial_workflow.plot_spatial_jsi'):
    output=run_spatial_study(raw,spatial,17);old=set((output/'cache').glob('*.npz'))
    self.assertEqual(len(old),4)
    same=run_spatial_study(raw,spatial,19);self.assertEqual(same,output)
    self.assertEqual(len(list((output/'cache').glob('*.npz'))),4)
    self.assertTrue(all(not p.exists() for p in old))
    set_profile('comparison');other=run_spatial_study(raw,spatial,17)
    self.assertNotEqual(other,output);self.assertEqual(len(list((output/'cache').glob('*.npz'))),4)
 def test_deletion_is_confined_to_owner(self):
  with tempfile.TemporaryDirectory() as tmp:
   owner=Path(tmp)/'owned';owner.mkdir()
   with self.assertRaises(ValueError):remove_owned(owner,owner)
   with self.assertRaises(ValueError):remove_owned(Path(tmp)/'elsewhere',owner)
