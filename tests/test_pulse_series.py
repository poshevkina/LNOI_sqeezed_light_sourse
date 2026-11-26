import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from linbo3.pulse_series import (default_windows,pulse_config,pulse_directory,
    run_pulse_series,pulse_saved_is_current,render_pulse_figures,plot_pulse_k)
from linbo3.mode_classification import CrossSection
from linbo3.spatial_workflow import prepare_spatial_study,spatial_directory
from test_scientific_workflow import write_saved_run
from test_spatial_workflow import classification_table


class PulseSeriesTests(unittest.TestCase):
    def test_plan_and_independent_cache_directories(self):
        plan=default_windows()
        self.assertEqual(set(plan),{250,500,750,1000,1250,1500,1750,2000})
        self.assertEqual(plan[250],.015);self.assertEqual(plan[2000],.0025)
        self.assertTrue(np.all(np.diff(list(plan.values()))<0))
        for points in [2,16,31.5]:
            with self.assertRaises(ValueError):pulse_config(300,.015,points)

    def test_numeric_cache_resume_and_render_never_call_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=write_saved_run(Path(temporary)/'run')
            spatial=prepare_spatial_study(run,classification_table())
            plan={250:default_windows()[250],500:default_windows()[500]}
            with patch('linbo3.pump_hybridization.read_cross_section',return_value=CrossSection(.5,.1,70)), \
                 patch('linbo3.pump_hybridization.classify_field',return_value=({'polarization_purity':.85},{})):
                results=run_pulse_series(run,spatial,plan,points=17,progress=False)
            self.assertNotEqual(results[250],results[500])
            self.assertFalse(any(p.name=='figures' for p in run.rglob('figures')))
            arrays=list(run.rglob('cache/*.npz'));self.assertEqual(len(arrays),8)
            times={p:p.stat().st_mtime_ns for p in arrays}
            for t,dw in plan.items():
                self.assertTrue(pulse_saved_is_current(run,spatial,pulse_config(t,dw,17)))
                settings=json.loads((results[t]/'settings.json').read_text())
                self.assertEqual(settings['pulse']['t_fs'],t)
            with patch('linbo3.sweep._compute_case',side_effect=AssertionError('Unexpected recalculation')), \
                 patch('linbo3.spdc.calculate_spectrum',side_effect=AssertionError('Unexpected solver')), \
                 patch('linbo3.pump_hybridization.track_modes',side_effect=AssertionError('Unexpected audit recalculation')), \
                 patch('matplotlib.pyplot.show'):
                index=run/'analysis/main/pulses/summary_all.csv'
                index_stamp=index.stat().st_mtime_ns
                second=run_pulse_series(run,spatial,plan,points=17,calculate=False,progress=False,write_index=False)
                self.assertEqual(index.stat().st_mtime_ns,index_stamp)
                figures=render_pulse_figures(run,spatial,250,plan[250],points=17,dpi=65)
                self.assertEqual(len(figures),3)
                self.assertTrue(all(p.is_file() for p in figures))
                plot_pulse_k(run,spatial,250,plan[250],points=17,dpi=65)
                with self.assertRaises(FileNotFoundError):render_pulse_figures(run,spatial,250,.014,points=17)
            self.assertEqual(second,results)
            self.assertEqual(times,{p:p.stat().st_mtime_ns for p in arrays})
            legacy=pulse_directory(run,spatial,pulse_config(2000,.0025,17))
            self.assertEqual(legacy,spatial_directory(run,17,6))
            self.assertFalse(pulse_saved_is_current(run,spatial,pulse_config(250,.014,17)))
            self.assertFalse(pulse_saved_is_current(run,spatial,pulse_config(250,plan[250],19)))
            summary=pd.read_csv(run/'analysis/main/pulses/summary_all.csv')
            self.assertEqual(len(summary),8)
            self.assertIn('pump_interval_hybrid',summary)


if __name__=='__main__':unittest.main()
