from .storage import analysis_root, remove_owned
"""Cached pulse-duration sweeps using the same spatial branches and hybrid audit."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .sweep import PulseConfig,run_sweep
from .spatial_workflow import (_identity,spatial_directory,spatial_saved_is_current,
    plot_spatial_jsi,plot_spatial_k)
from .cached_spectra import cached_case
from .pump_hybridization import (PumpHybridizationAudit,audit_pump_hybridization,validated_hybrid_flags)
from .marginal_spectra import plot_marginal_spectra


def default_windows():
    """Affine in 1/t, exactly DW(250)=.015 and DW(2000)=.0025."""
    durations=[250,500,750,1000,1250,1500,1750,2000]
    return {t:(.015 if t==250 else .0025 if t==2000 else
        .0025+(.015-.0025)*(1/t-1/2000)/(1/250-1/2000)) for t in durations}


def pulse_config(t_fs,dw,points):
    if (not np.isfinite([t_fs,dw,points]).all() or t_fs<=0 or dw<=0
            or int(points)!=points or points<3 or int(points)%2==0):
        raise ValueError('Нужны t>0, DW>0 и нечётное число точек >=3')
    t=float(t_fs)
    return PulseConfig(int(t) if t.is_integer() else t,float(dw),int(points))


def pulse_directory(run_dir,spatial,pulse):
    if pulse.t_fs==2000 and pulse.dw_half==.0025:
        return spatial_directory(run_dir,pulse.n_points,spatial.study.degrees[0])
    return analysis_root(run_dir)/'pulses'/f't_{pulse.t_fs:g}fs'


def pulse_saved_is_current(run_dir,spatial,pulse,output=None):
    output=Path(output) if output is not None else pulse_directory(run_dir,spatial,pulse)
    if not spatial_saved_is_current(run_dir,spatial,pulse.n_points,allow_cached_subset=True,output_dir=output):return False
    try:
        summary=pd.read_csv(output/'summary.csv',float_precision='round_trip')
        manifest=json.loads((Path(run_dir)/'config.json').read_text(encoding='utf-8-sig'))
        for width,mapping in spatial.study.pump_map.items():
            for label in mapping:
                row=summary.loc[summary.width_um.eq(width)&summary.pump_label.eq(label)]
                if (len(row)!=1 or float(row.iloc[0].t_fs)!=pulse.t_fs
                        or float(row.iloc[0].DW)!=pulse.dw_half or int(row.iloc[0].N_POINTS)!=pulse.n_points):return False
                cached_case(output/'cache',spatial,width,label,summary,manifest,pulse.n_points)
    except (OSError,ValueError,KeyError):return False
    return True


def load_hybridization(output,spatial):
    """Restore and validate flags without recalculating JSI or retracking fields."""
    output=Path(output)
    settings=json.loads((output/'pump_hybridization_settings.json').read_text(encoding='utf-8'))
    report=PumpHybridizationAudit(pd.read_csv(output/'pump_hybridization_cases.csv'),
        pd.read_csv(output/'pump_hybridization_samples.csv'),settings['identity'],
        [tuple(item) for item in settings['inputs']],settings['fraction'],settings['mixing_limit'],output)
    validated_hybrid_flags(report,output,spatial)
    return report


def ensure_hybridization(run_dir,spatial,output,pulse,*,fraction=.99,mixing_limit=.1):
    try:
        report=load_hybridization(output,spatial)
        if report.fraction==fraction and report.mixing_limit==mixing_limit:return report
    except (OSError,ValueError,KeyError):pass
    return audit_pump_hybridization(run_dir,spatial,output,points=pulse.n_points,
        fraction=fraction,mixing_limit=mixing_limit)


def run_pulse_series(run_dir,spatial,windows=None,*,points=1501,fraction=.99,mixing_limit=.1,
                     calculate=True,progress=True,write_index=True):
    """Store full JSI, K and field audit, but do not render/display any figures.

    Existing per-case numerical caches are reusable after interruption; summary and
    state='complete' are committed only after run_sweep has completed a duration.
    """
    run_dir=Path(run_dir);windows=default_windows() if windows is None else windows
    if not windows:raise ValueError('Задай хотя бы одну длительность и DW')
    if not 0<fraction<1 or not 0<mixing_limit<.5:raise ValueError('Проверь долю JSI и порог гибридизации')
    pulses=[pulse_config(t,dw,points) for t,dw in windows.items()]
    manifest=json.loads((run_dir/'config.json').read_text(encoding='utf-8-sig'))
    outputs={};all_rows=[]
    for pulse in pulses:
        output=pulse_directory(run_dir,spatial,pulse)
        current=pulse_saved_is_current(run_dir,spatial,pulse,output)
        if not current and not calculate:
            raise FileNotFoundError(f'Нет актуального расчёта t={pulse.t_fs:g} фс, DW=±{pulse.dw_half:g}; установи RUN_PULSE_SERIES=True')
        if progress:print(f'\nt={pulse.t_fs:g} фс | DW=±{pulse.dw_half:.7g} рад/фс'+(' | готовый кэш' if current else ''),flush=True)
        if not current:
            output.mkdir(parents=True,exist_ok=True)
            settings=dict(state='running',identity=_identity(run_dir,spatial,pulse.n_points),pulse=asdict(pulse))
            (output/'settings.json').write_text(json.dumps(settings),encoding='utf-8')
            sweep=run_sweep(spatial.study,pulse,output,length_m=manifest.get('length_m',.12),progress=progress)
            table=sweep.summary.merge(spatial.modes,on=['width_um','pump_label','pump_mode'],validate='many_to_one')
            table['any_hybrid']=table[['pump_hybrid','signal_hybrid','idler_hybrid']].any(axis=1)
            table.to_csv(output/'summary.csv',index=False)
            spatial.diagnostics.to_csv(output/'spatial_diagnostics.csv',index=False)
            settings['state']='complete'
            (output/'settings.json').write_text(json.dumps(settings),encoding='utf-8')
        report=ensure_hybridization(run_dir,spatial,output,pulse,fraction=fraction,mixing_limit=mixing_limit)
        outputs[pulse.t_fs]=output
        table=pd.read_csv(output/'summary.csv',float_precision='round_trip')
        allowed=spatial.modes[['width_um','pump_label']]
        table=table.merge(allowed,on=['width_um','pump_label'],validate='one_to_one')
        table=table.merge(report.cases[['width_um','pump_label','pump_interval_hybrid','checked_points','status']],
            on=['width_um','pump_label'],validate='one_to_one')
        all_rows.append(table)
        if progress:
            print(f'Случаев: {len(report.cases)}; гибридных: {int(report.cases.pump_interval_hybrid.sum())}; '
                f'проверенных точек COMSOL: {report.cases.checked_points.min()}–{report.cases.checked_points.max()}.',flush=True)
    if write_index:
        destination=analysis_root(run_dir)/'pulses'
        destination.mkdir(parents=True,exist_ok=True)
        pd.concat(all_rows,ignore_index=True).to_csv(destination/'summary_all.csv',index=False,encoding='utf-8-sig')
    if calculate:
        folder=analysis_root(run_dir)/'pulses'
        keep={f't_{p.t_fs:g}fs' for p in pulses if p.t_fs!=2000 or p.dw_half!=.0025}
        if folder.exists():
            for path in folder.glob('t_*fs'):
                if path.is_dir() and path.name not in keep:remove_owned(path,folder)
    return outputs


def plot_pulse_k(run_dir,spatial,t_fs,dw,*,points=1501,figsize=(7.2,6),dpi=150):
    pulse=pulse_config(t_fs,dw,points);output=pulse_directory(run_dir,spatial,pulse)
    if not pulse_saved_is_current(run_dir,spatial,pulse,output):raise FileNotFoundError('Сначала выполни расчёт этой длительности в разделе 7')
    report=load_hybridization(output,spatial)
    plot_spatial_k(output,spatial=spatial,figsize=figsize,dpi=dpi,hybridization=report,
        title=f'Число Шмидта · t={pulse.t_fs:g} фс\nDW=±{pulse.dw_half:.5g} рад/фс')
    return output/'schmidt_number_vs_width.png'


def render_pulse_figures(run_dir,spatial,t_fs,dw,*,points=1501,jsi=True,spectra=True,dpi=150):
    """Read existing full arrays only. Never start calculate_spectrum/run_sweep."""
    pulse=pulse_config(t_fs,dw,points);output=pulse_directory(run_dir,spatial,pulse)
    if not pulse_saved_is_current(run_dir,spatial,pulse,output):
        raise FileNotFoundError(f'Нет готовых JSI для {pulse.t_fs:g} фс: сначала выполни раздел 7')
    load_hybridization(output,spatial)
    title=f't={pulse.t_fs:g} фс · DW=±{pulse.dw_half:.5g} рад/фс'
    figures=[]
    if jsi:figures.append(plot_spatial_jsi(output,spatial,run_dir=run_dir,title=title))
    if spectra:figures.extend(plot_marginal_spectra(run_dir,spatial,output,points=points,dpi=dpi,title=title))
    return tuple(figures)


def plot_pulse_k_grid(run_dir,spatial,windows,*,points=1501,figsize=(24,11),dpi=70):
    """One figure, four columns; reuse the same curves and hybrid markers."""
    from matplotlib import pyplot as plt
    if not windows:raise ValueError('Нет длительностей')
    rows=(len(windows)+3)//4
    fig,axes=plt.subplots(rows,4,figsize=figsize,dpi=dpi,squeeze=False,layout='constrained')
    handles={}
    try:
        for ax,(duration,dw) in zip(axes.flat,windows.items()):
            pulse=pulse_config(duration,dw,points);output=pulse_directory(run_dir,spatial,pulse)
            if not pulse_saved_is_current(run_dir,spatial,pulse):raise FileNotFoundError(f'Нет расчёта {duration} фс')
            report=load_hybridization(output,spatial)
            plot_spatial_k(output,spatial=spatial,hybridization=report,ax=ax,
                title=f't={duration:g} фс · DW=±{dw:.5g}')
            for h,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,h)
            ax.get_legend().remove()
            ax.title.set_fontsize(13);ax.xaxis.label.set_size(12);ax.yaxis.label.set_size(12)
            ax.tick_params(labelsize=10)
        for ax in list(axes.flat)[len(windows):]:ax.set_visible(False)
        fig.legend(list(handles.values()),list(handles),loc='outside lower center',ncol=3,fontsize=12)
        fig.suptitle('Число Шмидта от ширины для разных длительностей',fontsize=18)
        destination=analysis_root(run_dir)/'pulses'
        destination.mkdir(exist_ok=True,parents=True)
        path=destination/'K_overview.png'
        fig.savefig(path,dpi=dpi);plt.show()
        return path
    finally:plt.close(fig)


def plot_k_duration_by_width(run_dir,spatial,windows,*,points=1501,figsize=(18,10),dpi=70):
    """K versus duration, three columns, physical mode labels and both flags."""
    from matplotlib import pyplot as plt
    from .spatial_workflow import label_key
    widths=list(spatial.study.widths)
    if not widths or not windows:raise ValueError('Нет ширин или длительностей')
    frames=[]
    for duration,dw in sorted(windows.items()):
        pulse=pulse_config(duration,dw,points);folder=pulse_directory(run_dir,spatial,pulse)
        if not pulse_saved_is_current(run_dir,spatial,pulse):raise FileNotFoundError(f'Нет расчёта {duration} фс')
        flags=load_hybridization(folder,spatial).cases
        frame=pd.read_csv(folder/'summary.csv',float_precision='round_trip')
        frame=frame.merge(flags[['width_um','pump_label','pump_interval_hybrid']],
            on=['width_um','pump_label'],validate='one_to_one')
        frames.append(frame)
    table=pd.concat(frames,ignore_index=True)
    labels=sorted(spatial.study.pump_labels,key=label_key)
    colors={label:plt.get_cmap('tab10')(i) for i,label in enumerate(labels)}
    fig,axes=plt.subplots((len(widths)+2)//3,3,figsize=figsize,dpi=dpi,squeeze=False,layout='constrained')
    handles={};durations=sorted(windows)
    try:
        for ax,width in zip(axes.flat,widths):
            for label in labels:
                part=table.loc[table.width_um.eq(width)&table.pump_label.eq(label)].set_index('t_fs').reindex(durations)
                if not part.K.notna().any():continue
                ax.plot(durations,part.K,'o-',label=label,color=colors[label],lw=1.8,ms=5)
                for column,color,size,legend,z in [
                    ('any_hybrid','crimson',65,'Hybrid >10% в центре',5),
                    ('pump_interval_hybrid','darkorange',28,'Накачка Hybrid в спектральном интервале',7)]:
                    marked=part[column].eq(True)&part.K.notna()
                    if marked.any():ax.scatter(part.index[marked],part.loc[marked,'K'],marker='D',color=color,s=size,label=legend,zorder=z)
            ax.set(title=f'w = {width:g} мкм',xlabel='Длительность импульса, фс',ylabel='Число Шмидта K',xticks=durations)
            ax.title.set_fontsize(15);ax.tick_params(labelsize=10);ax.grid(alpha=.25)
            for h,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,h)
        for ax in list(axes.flat)[len(widths):]:ax.set_visible(False)
        fig.legend(list(handles.values()),list(handles),loc='outside lower center',ncol=3,fontsize=12)
        fig.suptitle('Число Шмидта от длительности импульса',fontsize=18)
        folder=analysis_root(run_dir)/'pulses'
        folder.mkdir(exist_ok=True,parents=True)
        path=folder/'K_vs_duration.png'
        fig.savefig(path,dpi=dpi);plt.show();return path
    finally:plt.close(fig)
