"""Audit the tracked pump fields in a sum-frequency probability interval of JSI."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from .frequency import LIGHT_SPEED_M_PER_FS
from .modes import track_modes
from .saved_modes import field_path, load_pair
from .mode_classification import classify_field, read_cross_section
from .spatial_workflow import _identity, spatial_saved_is_current
from .cached_spectra import cached_case
from .spectral_projections import saved_sum_frequency, sum_frequency_distribution


def contribution_interval(omega_sum, mass, fraction=.99):
    """Equal-tail discrete quantiles; included bin centres contain >= fraction.

    No interpolation pretending to add spectral resolution between JSI bins.
    """
    x=np.asarray(omega_sum,dtype=float);p=np.asarray(mass,dtype=float)
    if (not np.isfinite(fraction) or not 0<fraction<1 or x.ndim!=1 or x.shape!=p.shape
            or len(x)<2 or not np.isfinite(x).all() or np.any(np.diff(x)<=0)
            or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0):
        raise ValueError('Нужны 0 < доля < 1 и корректное распределение суммы частот')
    p=p/p.sum();cdf=np.cumsum(p);tail=(1-fraction)/2
    lo=min(int(np.searchsorted(cdf,tail)),len(x)-1)
    hi=min(int(np.searchsorted(cdf,1-tail)),len(x)-1)
    return float(x[lo]),float(x[hi]),float(p[lo:hi+1].sum())


def _fingerprint(paths,base=None):
    return [(Path(os.path.relpath(Path(p).resolve(),Path(base).resolve())).as_posix()
             if base is not None else str(Path(p).resolve()),
             Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in paths]


@dataclass
class PumpHybridizationAudit:
    cases: pd.DataFrame
    samples: pd.DataFrame
    identity: dict
    inputs: list
    fraction: float
    mixing_limit: float
    output_dir: Path


def audit_pump_hybridization(run_dir,spatial,output,*,points=1501,fraction=.99,mixing_limit=.1,cache_dir=None):
    """Use cached JSI and existing tracked complex fields only; no solver calls."""
    if not np.isfinite(fraction) or not 0<fraction<1:raise ValueError('Нужно 0 < SPECTRAL_FRACTION < 1')
    if not np.isfinite(mixing_limit) or not 0<mixing_limit<.5:raise ValueError('Нужно 0 < MIXING_LIMIT < 0.5')
    run_dir,output=Path(run_dir),Path(output)
    if not spatial_saved_is_current(run_dir,spatial,points,allow_cached_subset=True,output_dir=Path(cache_dir).parent if cache_dir is not None else output):
        raise ValueError('Сначала рассчитай актуальные JSI/K в разделе 5')
    manifest=json.loads((run_dir/'config.json').read_text(encoding='utf-8-sig'))
    summary_path=(Path(cache_dir).parent if cache_dir is not None else output)/'summary.csv'
    summary=pd.read_csv(summary_path,float_precision="round_trip")
    geometry=read_cross_section(run_dir,manifest)
    paths=[summary_path];records=[];samples=[]
    cache=Path(cache_dir) if cache_dir is not None else output/'cache'
    constant=2*np.pi*LIGHT_SPEED_M_PER_FS*1e9
    for width,mapping in spatial.study.pump_map.items():
        directory=run_dir/f'w_{width:g}'
        _,raw=load_pair(directory,'PUMP',manifest['wavelengths_um']['PUMP'])
        paths.extend([field_path(directory, 'PUMP'),directory/'PUMP_raw.csv'])
        for label,block in mapping.items():
            part=summary.loc[summary.width_um.eq(width)&summary.pump_label.eq(label)]
            if len(part)!=1:raise ValueError(f'Неоднозначные данные JSI: w={width:g}, {label}')
            case=part.iloc[0]
            if int(case.N_POINTS)!=points:raise ValueError('Сетка JSI не соответствует STUDY_POINTS')
            path, pulse = cached_case(cache, spatial, width, label, part, manifest, points)
            paths.append(path)
            with np.load(path,allow_pickle=False) as saved:
                omega,mass=saved_sum_frequency(saved)
            low,high,actual=contribution_interval(omega,mass,fraction)
            lambda_min,lambda_max=constant/high,constant/low
            meta=spatial.modes.loc[spatial.modes.width_um.eq(width)&spatial.modes.pump_label.eq(label)].iloc[0]
            tracked=track_modes(raw,band='PUMP',anchor_um=manifest['pump_wavelength_um'],
                anchor_solnums=[int(meta.anchor_solnum)],**spatial.tracking)
            rows=sorted(tracked.rows,key=lambda row:row['wavelength_um'])
            model=spatial.study.raw[width]['PUMP'][block]
            if (not np.array_equal(tracked.modes[0].wavelength_um,model.wavelength_um)
                    or not np.array_equal(tracked.modes[0].neff,model.neff)):
                raise ValueError(f'Отслеженная ветвь изменилась: w={width:g}, {label}; выполни раздел 4')
            checked=[];errors=0
            for item in rows:
                wavelength=item['wavelength_um']*1000
                if not lambda_min-1e-9<=wavelength<=lambda_max+1e-9:continue
                matches=np.flatnonzero((raw.outer_solnum==item['outer_solnum'])&(raw.solnum==item['solnum'])
                    & np.isclose(raw.wavelength_um,item['wavelength_um'],rtol=0,atol=1e-12))
                record=dict(width_um=width,pump_label=label,wavelength_nm=wavelength,
                    outer_solnum=item['outer_solnum'],solnum=item['solnum'],overlap=item['overlap'],
                    minority_fraction=np.nan,hybrid=False,status='ok',message='')
                try:
                    if len(matches)!=1:raise ValueError('Неоднозначный идентификатор поля')
                    classified,_=classify_field(raw.coordinates,raw.field[matches[0]],width,geometry)
                    minority=1-classified['polarization_purity']
                    record.update(minority_fraction=minority,hybrid=bool(minority>mixing_limit+1e-12))
                    checked.append(record)
                except ValueError as error:
                    errors+=1;record.update(status='field_error',message=str(error))
                samples.append(record)
            any_hybrid=any(item['hybrid'] for item in checked)
            grid=np.unique(raw.wavelength_um)*1000
            covered=bool(lambda_min>=grid[0]-1e-9 and lambda_max<=grid[-1]+1e-9)
            status=('field_error' if errors else 'no_samples' if not checked else
                    'outside_comsol_range' if not covered else 'single_sample' if len(checked)==1 else 'sampled')
            records.append(dict(width_um=width,pump_label=label,omega_min=low,omega_max=high,
                lambda_min_nm=lambda_min,lambda_max_nm=lambda_max,requested_fraction=fraction,
                included_fraction=actual,mixing_limit=mixing_limit,checked_points=len(checked),
                hybrid_points=sum(item['hybrid'] for item in checked),pump_interval_hybrid=any_hybrid,
                max_minority_fraction=max((item['minority_fraction'] for item in checked),default=np.nan),
                status=status,within_comsol_range=covered,t_fs=pulse.t_fs,DW=pulse.dw_half,N_POINTS=points))
    output.mkdir(exist_ok=True,parents=True)
    sample_columns=['width_um','pump_label','wavelength_nm','outer_solnum','solnum','overlap',
        'minority_fraction','hybrid','status','message']
    report=PumpHybridizationAudit(pd.DataFrame(records),pd.DataFrame(samples,columns=sample_columns),_identity(run_dir,spatial,points),
        _fingerprint(paths,base=output),fraction,mixing_limit,output)
    report.cases.to_csv(output/'pump_hybridization_cases.csv',index=False,encoding='utf-8-sig')
    report.samples.to_csv(output/'pump_hybridization_samples.csv',index=False,encoding='utf-8-sig')
    (output/'pump_hybridization_settings.json').write_text(json.dumps(dict(version=1,
        identity=report.identity,inputs=report.inputs,input_paths_base="report_dir",fraction=fraction,mixing_limit=mixing_limit,
        rule='equal_tail_discrete_sum_frequency_quantiles; sampled_fields_only'),ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def validated_hybrid_flags(report,output,spatial):
    """Never apply flags from a different study or changed numerical cache."""
    output=Path(output)
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    # Allow a current subset of an older sweep, just as spatial_saved_is_current does.
    old=settings.get('identity',{});current=report.identity
    keys=('source_signature','points','degree','tracking','signal_label','idler_label')
    if (any(old.get(key)!=current.get(key) for key in keys)
            or not all(mode in old.get('modes',[]) for mode in current.get('modes',[]))
            or spatial.study.degrees[0]!=current.get('degree')
            or spatial.tracking!=current.get('tracking')
            or json.loads(spatial.modes.to_json(orient='records'))!=current.get('modes')):
        raise ValueError('Проверка гибридизации устарела: выполни раздел 5.2')
    try:
        # Old in-memory reports remain readable; new reports store relative paths.
        relative=all(not Path(item[0]).is_absolute() for item in report.inputs)
        unchanged=_fingerprint([output/item[0] for item in report.inputs],
            base=output if relative else None)==report.inputs
    except OSError:unchanged=False
    if not unchanged:raise ValueError('Данные JSI/полей изменились: выполни раздел 5.2')
    if (output.resolve()/'summary.csv').resolve()!=(output/Path(report.inputs[0][0])).resolve():
        raise ValueError('Проверка гибридизации относится к другому расчёту: выполни раздел 5.2')
    return report.cases
