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
 def test_comsol_runs_preserve_history_and_reuse_pairs(self):
  from linbo3.comsol import MATERIAL_TABLES
  from linbo3.saved_modes import load_pair
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'model.mph').write_bytes(b'model')
   for table in MATERIAL_TABLES.values():(root/table).write_text('material')
   cfg=dict(model='model.mph',output_dir='results',widths_um=[.5,.7],geometry={},
    bands={'PUMP':dict(start_um=.7,stop_um=.85,points=9),'IS':dict(start_um=1.4,stop_um=1.7,points=9)},
    modes=None,cores=2,field_grid=[2,2],pump_wavelength_um=.775)
   path=root/'config.json'
   def solve(model,wavelengths,**kw):
    kw['log'].write_text('complete');return modal_fixture('PUMP' if wavelengths[0]<1 else 'IS')
   with patch('linbo3.comsol_models.single.model.solve_modal_fields',side_effect=solve) as solver:
    path.write_text(json.dumps(cfg));first=run(path,stage='comsol');self.assertEqual(solver.call_count,4)
    self.assertEqual(run(path,stage='comsol'),first);self.assertEqual(solver.call_count,4)
    cfg['widths_um']=[.5,.7,.9];path.write_text(json.dumps(cfg))
    second=run(path,stage='comsol');self.assertEqual(solver.call_count,6)
    self.assertNotEqual(first,second)
    self.assertTrue((first/'w_0.5'/'PUMP_raw.csv').is_file())
    self.assertEqual(len(list((root/'field_maps').glob('*/field.npz'))),6)
    self.assertFalse(list((root/'results').rglob('*.npz')))
    a=json.loads((first/'w_0.5'/'PUMP_field.json').read_text(encoding='utf-8'))
    b=json.loads((second/'w_0.5'/'PUMP_field.json').read_text(encoding='utf-8'))
    self.assertEqual(a['field_id'],b['field_id'])
    self.assertIn('w0.5_PUMP_0.7-0.85um_9pts_2x2',a['field_id'])
    _,raw=load_pair(second/'w_0.5','PUMP',a['wavelengths_um'])
    self.assertEqual(raw.field.shape,(27,3,4))
    (root/'n_SiO2.txt').write_text('new material')
    third=run(path,stage='comsol');self.assertEqual(solver.call_count,12)
    self.assertNotEqual(second,third)
    self.assertTrue((second/'status.json').is_file())
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
