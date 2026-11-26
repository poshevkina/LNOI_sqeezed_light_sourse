import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from test_scientific_workflow import write_saved_run
from linbo3.spatial_workflow import prepare_spatial_study, run_spatial_study, spatial_saved_is_current


def classification_table():
    rows=[]
    for width in (.5,.7):
        for band in ('IS','PUMP'):
            # Reverse TE/TM neff ordering between widths; labels, not ranks, must win.
            for solnum,label in enumerate(('TE0','TM0') if width==.5 else ('TM0','TE0'),1):
                rows.append(dict(width_um=width,band=band,solnum=solnum,status='confident',
                    candidate_label=label,reason='',polarization_purity=.85 if band=='PUMP' and label=='TE0' else .99,
                    te_fraction=.85 if label=='TE0' else .01))
    return pd.DataFrame(rows)


class SpatialWorkflow(unittest.TestCase):
    def test_labels_choose_fields_despite_reversed_neff_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=write_saved_run(Path(temporary)/'run')
            spatial=prepare_spatial_study(run,classification_table())
            meta=spatial.modes
            self.assertEqual(int(meta.loc[meta.width_um.eq(.5)&meta.pump_label.eq('TE0'),'anchor_solnum'].iloc[0]),1)
            self.assertEqual(int(meta.loc[meta.width_um.eq(.7)&meta.pump_label.eq('TE0'),'anchor_solnum'].iloc[0]),2)
            self.assertGreater(spatial.study.raw[.5]['IS'][0].neff[4],spatial.study.raw[.5]['IS'][1].neff[4])
            self.assertLess(spatial.study.raw[.7]['IS'][0].neff[4],spatial.study.raw[.7]['IS'][1].neff[4])

    def test_summary_has_spatial_labels_and_hybrid_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=write_saved_run(Path(temporary)/'run')
            spatial=prepare_spatial_study(run,classification_table())
            output=run_spatial_study(run,spatial,17)
            summary=pd.read_csv(output/'summary.csv')
            self.assertEqual(set(summary.pump_label),{'TE0','TM0'})
            self.assertTrue(summary.loc[summary.pump_label.eq('TE0'),'pump_hybrid'].all())
            self.assertFalse(summary.loc[summary.pump_label.eq('TM0'),'pump_hybrid'].any())
            self.assertTrue(spatial_saved_is_current(run,spatial,17))
            other=prepare_spatial_study(run,classification_table(),signal_label='TE0',idler_label='TM0')
            self.assertFalse(spatial_saved_is_current(run,other,17))

    def test_more_than_two_is_modes_excludes_the_whole_width(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            run=write_saved_run(Path(temporary)/'run')
            path=run/'w_0.5/IS_raw.npz'
            with np.load(path) as saved:data={key:saved[key] for key in saved.files}
            data['neff'].imag[data['solnum']==3]=1e-9
            np.savez_compressed(path,**data)
            table=pd.read_csv(run/'w_0.5/IS_raw.csv');table['neff_imag']=data['neff'].imag
            table.to_csv(run/'w_0.5/IS_raw.csv',index=False)
            spatial=prepare_spatial_study(run,classification_table())
            self.assertEqual(set(spatial.study.widths),{.7})
            self.assertTrue(spatial.diagnostics.message.str.contains('больше двух').any())

    def test_duplicate_labels_are_rejected_instead_of_ranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=write_saved_run(Path(temporary)/'run')
            table=classification_table()
            table.loc[table.width_um.eq(.5)&table.band.eq('PUMP'),'candidate_label']='TE0'
            spatial=prepare_spatial_study(run,table)
            self.assertNotIn(.5,spatial.study.raw)
            self.assertTrue(spatial.diagnostics.message.str.contains('Повторная').any())


if __name__=='__main__':unittest.main()
