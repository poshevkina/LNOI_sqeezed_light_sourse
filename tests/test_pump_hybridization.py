import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from linbo3.frequency import LIGHT_SPEED_M_PER_FS
from linbo3.mode_classification import CrossSection
from linbo3.pump_hybridization import (sum_frequency_distribution,contribution_interval,
    audit_pump_hybridization,validated_hybrid_flags)


class SumFrequencyTests(unittest.TestCase):
    def test_anticorrelated_broad_marginals_have_narrow_sum(self):
        omega=np.linspace(1,2,31)
        jsi=np.fliplr(np.eye(31))
        x,p=sum_frequency_distribution(omega,jsi)
        self.assertEqual(np.count_nonzero(p),1)
        low,high,actual=contribution_interval(x,p)
        self.assertAlmostEqual(low,3);self.assertAlmostEqual(high,3)
        self.assertEqual(actual,1)

    def test_separable_gaussian_sum_variance_and_discrete_coverage(self):
        omega=np.linspace(1,2,701)
        a=np.exp(-.5*((omega-1.5)/.04)**2)
        b=np.exp(-.5*((omega-1.5)/.07)**2)
        x,p=sum_frequency_distribution(omega,np.outer(b,a))
        mean=np.sum(x*p);variance=np.sum((x-mean)**2*p)
        self.assertAlmostEqual(mean,3,places=10)
        self.assertAlmostEqual(variance,.04**2+.07**2,places=10)
        low,high,actual=contribution_interval(x,p,.99)
        self.assertGreaterEqual(actual,.99)
        self.assertAlmostEqual(3-low,high-3,places=10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):sum_frequency_distribution([1,2,4],np.ones((3,3)))
        with self.assertRaises(ValueError):sum_frequency_distribution([1,2,3],np.zeros((3,3)))
        for fraction in [0,1,np.nan]:
            with self.assertRaises(ValueError):contribution_interval([1,2],[1,1],fraction)

    def test_sampled_fields_track_ids_and_reject_stale_report(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';out=root/'output'
            run.mkdir();out.mkdir();(out/'cache').mkdir()
            widthdir=run/'w_0.5';widthdir.mkdir()
            for name in ['PUMP_raw.csv','PUMP_raw.npz']:(widthdir/name).write_bytes(b'input')
            manifest=dict(pump_wavelength_um=.775,wavelengths_um={'PUMP':[.7745,.775,.7755]})
            (run/'config.json').write_text(json.dumps(manifest))
            summary=pd.DataFrame([dict(width_um=.5,pump_label='TM0',t_fs=500,DW=.0025,N_POINTS=31)])
            summary.to_csv(out/'summary.csv',index=False)
            mode=SimpleNamespace(wavelength_um=np.array([.7745,.775,.7755]),neff=np.array([2.1,2.2,2.3]))
            metadata=pd.DataFrame([dict(width_um=.5,pump_label='TM0',anchor_solnum=4)])
            identity=dict(source_signature='abc',points=31,degree=6,tracking={},signal_label='TM0',
                idler_label='TE0',modes=json.loads(metadata.to_json(orient='records')))
            (out/'settings.json').write_text(json.dumps(dict(identity=identity)))
            study=SimpleNamespace(pump_map={.5:{'TM0':0}},degrees=(6,),raw={.5:{'PUMP':{0:mode}}})
            spatial=SimpleNamespace(study=study,modes=metadata,tracking={})
            # Real complex fields, including an unrelated hybrid distractor at the centre.
            x=np.linspace(-.6,.6,31);y=np.linspace(-.1,.6,25)
            xx,yy=np.meshgrid(x,y);shape=np.exp(-((xx/.2)**2+((yy-.25)/.15)**2)).ravel()
            fields=[]
            for minority in [.4,.02,.4,.45]:
                fields.append(np.array([np.sqrt(minority)*shape*1j,np.sqrt(1-minority)*shape,np.zeros_like(shape)]))
            raw=SimpleNamespace(coordinates=np.array([xx.ravel(),yy.ravel()])*1e-6,
                field=np.array(fields),wavelength_um=np.array([.7745,.775,.7755,.775]),
                outer_solnum=np.array([1,2,3,2]),solnum=np.array([9,4,8,99]))
            rows=[dict(wavelength_um=wl,outer_solnum=i+1,solnum=s,overlap=.99)
                for i,(wl,s) in enumerate(zip(mode.wavelength_um,[9,4,8]))]
            tracked=SimpleNamespace(rows=rows,modes={0:mode})
            cache=out/'cache'/'case.npz'
            center=2*np.pi*LIGHT_SPEED_M_PER_FS*1e6/.775
            omega=np.linspace(center/2-.0025,center/2+.0025,31)
            narrow=np.zeros((31,31));narrow[15,15]=1
            with patch('linbo3.pump_hybridization.spatial_saved_is_current',return_value=True), \
                 patch('linbo3.pump_hybridization.read_cross_section',return_value=CrossSection(.5,.1,70)), \
                 patch('linbo3.pump_hybridization.load_pair',return_value=(None,raw)), \
                 patch('linbo3.pump_hybridization.track_modes',return_value=tracked) as track, \
                 patch('linbo3.pump_hybridization._cache_path',return_value=cache) as locate, \
                 patch('linbo3.pump_hybridization._identity',return_value=identity):
                np.savez(cache,w_grid=omega,IMatrix=narrow)
                report=audit_pump_hybridization(run,spatial,out,points=31)
                self.assertFalse(report.cases.iloc[0].pump_interval_hybrid)
                self.assertEqual(report.cases.iloc[0].checked_points,1)
                self.assertEqual(report.cases.iloc[0].status,'single_sample')
                self.assertEqual(report.samples.solnum.tolist(),[4])
                self.assertEqual(track.call_args.kwargs['anchor_solnums'],[4])
                self.assertEqual(locate.call_args.args[5].t_fs,500)
                validated_hybrid_flags(report,out,spatial)
                np.savez(cache,w_grid=omega,IMatrix=np.ones((31,31)))
                with self.assertRaises(ValueError):validated_hybrid_flags(report,out,spatial)
                broad=audit_pump_hybridization(run,spatial,out,points=31)
                self.assertTrue(broad.cases.iloc[0].pump_interval_hybrid)
                self.assertEqual(broad.cases.iloc[0].hybrid_points,2)
                self.assertEqual(broad.cases.iloc[0].checked_points,3)
                self.assertEqual(broad.samples.solnum.tolist(),[9,4,8])
                # A well-defined spectral interval can contain no COMSOL samples.
                np.savez(cache,w_grid=omega+.02,IMatrix=narrow)
                absent=audit_pump_hybridization(run,spatial,out,points=31)
                self.assertEqual(absent.cases.iloc[0].status,'no_samples')
                self.assertEqual(absent.cases.iloc[0].checked_points,0)
                self.assertFalse(absent.cases.iloc[0].pump_interval_hybrid)
                self.assertEqual(len(pd.read_csv(out/'pump_hybridization_samples.csv')),0)


if __name__=='__main__':unittest.main()
