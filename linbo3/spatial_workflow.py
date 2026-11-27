from .storage import analysis_root
"""Follow spatially classified central modes; never match widths by neff rank."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .frequency import um_to_omega
from .interpolation import fit_dispersion
from .interpolation_study import source_signature
from .modes import track_modes, ModeTrackingError, selection_masks
from .saved_modes import load_pair
from .polynomial import DispersionStudy
from .study_plots import StudyPlots
from .sweep import PulseConfig, run_sweep
from .artifacts import write_json


def label_key(label):
    return (0 if label.startswith('TM') else 1, int(label[2:]))


@dataclass
class SpatialStudy:
    study: DispersionStudy
    modes: pd.DataFrame
    diagnostics: pd.DataFrame
    classification: pd.DataFrame
    tracking: dict
    signal_label: str
    idler_label: str
    branches: pd.DataFrame


def prepare_spatial_study(run_dir, classification, *, degree=6, tracking=None,
                          signal_label='TM0', idler_label='TE0'):
    """Explicitly select signal/idler by name; independently follow each PUMP field."""
    run_dir=Path(run_dir)
    manifest=json.loads((run_dir/'config.json').read_text(encoding='utf-8-sig'))
    thresholds=dict(imag_limit=1e-6,min_overlap=.9,min_margin=.1);thresholds.update(tracking or {})
    raw_modes={};mapping={};meta=[];diagnostics=[];branches=[]
    for width in sorted(manifest['widths_um']):
        data={};rows={}
        for band in ('IS','PUMP'):
            table=classification.loc[classification.width_um.eq(width)&classification.band.eq(band)].copy()
            table=table.loc[table.status.ne('excluded') & table.candidate_label.notna()]
            # Mixed polarization is allowed and marked; unresolved shape labels are not guessed.
            table=table.loc[table.reason.fillna('').isin(['','mixed_polarization'])]
            duplicates=table.candidate_label.duplicated(keep=False)
            for label in table.loc[duplicates,'candidate_label']:
                diagnostics.append(dict(width_um=width,band=band,label=label,message='Повторная пространственная метка'))
            table=table.loc[~duplicates]
            rows[band]={row['candidate_label']:row for row in table.to_dict('records')}
            try: _,data[band]=load_pair(run_dir/f'w_{width:g}',band,manifest['wavelengths_um'][band])
            except (OSError,ValueError,KeyError) as error:
                diagnostics.append(dict(width_um=width,band=band,label='',message=str(error)))
        if not all(b in data for b in ('IS','PUMP')):continue
        eligible=selection_masks(data['IS'],thresholds['imag_limit'])[2]
        central=np.isclose(data['IS'].wavelength_um,2*manifest['pump_wavelength_um'],atol=1e-10,rtol=0)
        count=int(np.count_nonzero(eligible & central))
        if count>2:
            diagnostics.append(dict(width_um=width,band='IS',label='',message=f'Исключена ширина: {count} допустимых IS-мод в центре (больше двух)'))
            continue
        if signal_label==idler_label or any(label not in rows['IS'] for label in (signal_label,idler_label)):
            diagnostics.append(dict(width_um=width,band='IS',label='',message='Нет однозначной выбранной пары signal/idler'))
            continue
        try:
            pair=track_modes(data['IS'],band='IS',anchor_um=2*manifest['pump_wavelength_um'],
                anchor_solnums=[int(rows['IS'][label]['solnum']) for label in (signal_label,idler_label)],**thresholds)
        except ModeTrackingError as error:
            diagnostics.append(dict(width_um=width,band='IS',label='',message=str(error)));continue
        pumps={};labels={};width_meta=[];pump_rows={};occupied={};collisions=set()
        for label in sorted(rows['PUMP'],key=label_key):
            row=rows['PUMP'][label]
            try:
                tracked=track_modes(data['PUMP'],band='PUMP',anchor_um=manifest['pump_wavelength_um'],
                    anchor_solnums=[int(row['solnum'])],**thresholds)
            except ModeTrackingError as error:
                diagnostics.append(dict(width_um=width,band='PUMP',label=label,message=str(error)));continue
            block=len(pumps);pumps[block]=tracked.modes[0];labels[label]=block
            pump_rows[label]=[dict(item, width_um=width, mode=block, spatial_label=label)
                              for item in tracked.rows]
            for item in tracked.rows:
                identifier=(item['outer_solnum'],item['solnum'])
                if identifier in occupied:
                    collisions.update((occupied[identifier],label))
                occupied[identifier]=label
            width_meta.append(dict(width_um=width,pump_label=label,pump_mode=block,
                anchor_solnum=int(row['solnum']),pump_hybrid=row['polarization_purity']<.9-1e-12,
                signal_label=signal_label,idler_label=idler_label,
                signal_hybrid=rows['IS'][signal_label]['polarization_purity']<.9-1e-12,
                idler_hybrid=rows['IS'][idler_label]['polarization_purity']<.9-1e-12,
                pump_te_fraction=row['te_fraction']))
        if collisions:
            for label in collisions:
                pumps.pop(labels.pop(label),None)
                diagnostics.append(dict(width_um=width,band='PUMP',label=label,message='Две пространственные ветви назначены одному решению; обе отклонены'))
            width_meta=[item for item in width_meta if item['pump_label'] not in collisions]
        if pumps:
            branches.extend(dict(item, width_um=width, spatial_label=(signal_label,idler_label)[item['mode']])
                            for item in pair.rows)
            for label in labels:
                branches.extend(pump_rows[label])
            raw_modes[width]={'IS':pair.modes,'PUMP':pumps};mapping[width]=labels;meta.extend(width_meta)
    if not raw_modes:raise ValueError('Нет подтверждённых пространственных мод для выбранной пары. Проверь классификацию и пороги.')
    fits={};errors=[]
    for width,bands in raw_modes.items():
        fits[width]={degree:{band:{} for band in bands}}
        for band,modes in bands.items():
            for block,mode in modes.items():
                poly=fit_dispersion(f'poly{degree}',mode.wavelength_um,mode.neff)
                fits[width][degree][band][block]=poly
                residual=poly(um_to_omega(mode.wavelength_um))-mode.neff
                errors.append(dict(width_um=width,degree=degree,branch=band,mode=block,RMSE=np.sqrt(np.mean(residual**2))))
    metadata=pd.DataFrame(meta)
    study=DispersionStudy(raw_modes,fits,pd.DataFrame(errors),metadata,mapping,(degree,),0,1,manifest['pump_wavelength_um'])
    issues = pd.DataFrame(diagnostics, columns=['width_um', 'band', 'label', 'message'])
    return SpatialStudy(study, metadata, issues, classification, thresholds, signal_label, idler_label,
                        pd.DataFrame(branches))


def spatial_directory(run_dir, points=1501, degree=6):
    return analysis_root(run_dir)/"spatial"


def _identity(run_dir,spatial,points):
    return dict(version=2,is_mode_policy="at_most_two_at_anchor",source_signature=source_signature(run_dir),points=points,
        degree=spatial.study.degrees[0],widths_um=json.loads((Path(run_dir)/"config.json").read_text(encoding="utf-8-sig"))["widths_um"],tracking=spatial.tracking,signal_label=spatial.signal_label,
        idler_label=spatial.idler_label,all_pump_labels=sorted(spatial.classification.loc[spatial.classification.band.eq('PUMP') & spatial.classification.candidate_label.fillna('').str.match(r'^(TE|TM)\d+$'), 'candidate_label'].unique().tolist(),key=label_key),modes=json.loads(spatial.modes.to_json(orient='records')))


def spatial_saved_is_current(run_dir,spatial,points=1501,*,allow_cached_subset=False,output_dir=None):
    output=Path(output_dir) if output_dir is not None else spatial_directory(run_dir,points,spatial.study.degrees[0]);path=output/'settings.json'
    if not path.is_file() or not (output/'summary.csv').is_file():return False
    saved=json.loads(path.read_text(encoding='utf-8'))
    current=_identity(run_dir,spatial,points)
    if saved.get('state')!='complete':return False
    if saved.get('identity')==current:return True
    if not allow_cached_subset:return False
    old=saved.get('identity',{})
    keys=('source_signature','points','degree','tracking','signal_label','idler_label')
    return (all(old.get(key)==current.get(key) for key in keys)
            and all(mode in old.get('modes',[]) for mode in current['modes']))


def plot_spatial_dispersion(spatial,output,*,show=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for width,bands in spatial.study.raw.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for row,band in enumerate(('PUMP','IS')):
            labels=spatial.study.pump_map[width] if band=='PUMP' else {spatial.signal_label:0,spatial.idler_label:1}
            for label,block in labels.items():
                mode=bands[band][block];poly=spatial.study.fits[width][spatial.study.degrees[0]][band][block]
                x=um_to_omega(mode.wavelength_um);dense=np.linspace(x.min(),x.max(),600)
                info=spatial.classification.loc[spatial.classification.width_um.eq(width)&spatial.classification.band.eq(band)&spatial.classification.candidate_label.eq(label)]
                hybrid=not info.empty and float(info.iloc[0].polarization_purity)<.9-1e-12
                for order in (0,1,2):
                    axes[row,order].plot(dense,poly.deriv(order)(dense),label=label+(' · Hybrid' if hybrid else ''))
                    axes[row,order].set(xlabel='ω, рад/фс',ylabel=('n_eff','dn_eff/dω, фс/рад','d²n_eff/dω², фс²/рад²')[order],title=band)
                    axes[row,order].grid(alpha=.2)
                axes[row,0].scatter(x,mode.neff,s=8)
                if hybrid:
                    center=float(um_to_omega(spatial.study.pump_wavelength_um*(1 if band=='PUMP' else 2)))
                    axes[row,0].plot(center,poly(center),'D',color='crimson',ms=7)
            for ax in axes[row]:ax.legend(fontsize=8)
        fig.suptitle(f'w={width:g} мкм · степень {spatial.study.degrees[0]} · пространственные моды')
        fig.savefig(output/f'disp_width_{width:g}.png',dpi=140)
        if show:
            plt.show()
        plt.close(fig)


def write_spatial_sweep(run_dir, spatial, pulse, output, *, cache_root=None, progress=True):
    """Calculate one pulse in an explicit directory using the shared SPDC cache."""
    output = Path(output)
    manifest = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8-sig"))
    identity = _identity(run_dir, spatial, pulse.n_points)
    sweep = run_sweep(spatial.study, pulse, output, length_m=manifest.get("length_m", .12),
                      cache_root=cache_root, progress=progress)
    summary = sweep.summary.merge(spatial.modes, on=["width_um", "pump_label", "pump_mode"],
                                  validate="many_to_one")
    summary["any_hybrid"] = summary[["pump_hybrid", "signal_hybrid", "idler_hybrid"]].any(axis=1)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(output / "settings.json", dict(state="complete", identity=identity))
    return sweep, summary


def run_spatial_study(run_dir,spatial,points=1501):
    output = spatial_directory(run_dir, points, spatial.study.degrees[0])
    sweep, _ = write_spatial_sweep(run_dir, spatial, PulseConfig(2000, .0025, points), output)
    spatial.diagnostics.to_csv(output / "spatial_diagnostics.csv", index=False)
    spatial.classification.to_csv(output / "classification.csv", index=False)
    plot_spatial_jsi(output, spatial, cases=sweep.cases)
    return output


def plot_spatial_jsi(output, spatial, *, cases=None, run_dir=None, title=None):
    """Draw grouped width rows; can redraw existing spectra without calculating them."""
    output=Path(output)
    degree=spatial.study.degrees[0]
    if cases is None:
        from .sweep import _cache_path
        settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
        points=settings['identity']['points']
        run_dir=Path(run_dir) if run_dir is not None else output.parents[1]
        manifest=json.loads((run_dir/'config.json').read_text(encoding='utf-8-sig'))
        from .cached_spectra import cached_case
        summary=pd.read_csv(output/"summary.csv",float_precision="round_trip")
        cases={}
        for width,mapping in spatial.study.pump_map.items():
            cases[width]={degree:{}}
            for label,block in mapping.items():
                path,pulse=cached_case(output/'cache',spatial,width,label,summary,manifest,points)
                if not path.is_file():raise FileNotFoundError('Нет файла JSI. Выполни раздел 5 с RUN_STUDY=True: '+str(path))
                with np.load(path,allow_pickle=False) as saved:
                    cases[width][degree][label]=dict(cache_path=path,K=float(saved['K']))
    plots=StudyPlots(spatial.study,output/'figures',show=False,max_plot_points=451)
    widths=list(spatial.study.widths);labels=sorted(spatial.study.pump_labels,key=label_key)
    fig=plt.figure(figsize=(max(6.0,3.8*len(labels)+.7),4.5*len(widths)+.4),layout='constrained')
    try:
        rows=fig.subfigures(len(widths),1,squeeze=False,hspace=.06)
        for row,width in enumerate(widths):
            subfig=rows[row,0]
            subfig.set_facecolor('#f1f4f7' if row%2==0 else '#ffffff')
            subfig.suptitle(f'w = {width:g} мкм',fontsize=17,fontweight='bold')
            axes=subfig.subplots(1,len(labels),squeeze=False)[0]
            mesh=None
            for ax,label in zip(axes,labels):
                case=cases[width][degree].get(label)
                if case is None:ax.set_visible(False);continue
                mesh=plots.jsi_panel(ax,case)
                ax.set_box_aspect(1)  # Квадратная область каждой отдельной JSI.
                metadata=spatial.modes.loc[spatial.modes.width_um.eq(width)&spatial.modes.pump_label.eq(label)].iloc[0]
                hybrid=bool(metadata.pump_hybrid or metadata.signal_hybrid or metadata.idler_hybrid)
                ax.set_title(label+(' · Hybrid' if hybrid else '')+f'\nK = {float(case["K"]):.4f}',fontsize=12,color='crimson' if hybrid else 'black')
                ax.tick_params(labelsize=10)
                ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11)
            if mesh is not None:
                colorbar=subfig.colorbar(mesh,ax=axes.tolist(),label='JSI',shrink=.8,pad=.02)
                colorbar.ax.tick_params(labelsize=10)
        fig.suptitle(title or f'JSI · signal {spatial.signal_label}, idler {spatial.idler_label}',fontsize=16)
        (output/'figures').mkdir(exist_ok=True)
        fig.savefig(output/'figures/jsi_overview.png',dpi=140)
    finally:plt.close(fig)
    return output/'figures/jsi_overview.png'


def plot_spatial_k(output, spatial=None, *, figsize=(7.2,6), dpi=150, hybridization=None, title=None, ax=None, show=True):
    size=np.asarray(figsize,dtype=float)
    if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError("figsize должен содержать положительные ширину и высоту в дюймах")
    if not np.isscalar(dpi) or not np.isfinite(dpi) or dpi<=0:
        raise ValueError("dpi должен быть положительным числом")
    output=Path(output);table=pd.read_csv(output/'summary.csv',float_precision='round_trip')
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    # Only widths admitted by the current two-IS-mode rule enter the plot.
    widths=list(spatial.study.widths) if spatial is not None else sorted(table.width_um.unique())
    table=table.loc[table.width_um.isin(widths)].copy()
    labels=sorted(table.pump_label.unique(),key=label_key)
    if table.empty:raise ValueError('Нет рассчитанных K для допустимых ширин')
    interval_flags=None
    if hybridization is not None:
        if spatial is None:raise ValueError('Для проверки гибридизации нужен spatial')
        from .pump_hybridization import validated_hybrid_flags
        interval_flags=validated_hybrid_flags(hybridization,output,spatial)
    standalone=ax is None
    if standalone:fig,ax=plt.subplots(figsize=tuple(size),dpi=dpi,layout='constrained')
    else:fig=ax.figure
    for label in labels:
        part=table.loc[table.pump_label.eq(label)].set_index('width_um').reindex(widths)
        line,=ax.plot(widths,part.K,'o-',label=label,lw=1.8,ms=6)
        hybrid=part.any_hybrid.fillna(False).astype(bool)
        if hybrid.any():ax.scatter(part.index[hybrid],part.loc[hybrid,'K'],marker='D',color='crimson',s=65,zorder=5)
        if interval_flags is not None:
            flagged=interval_flags.loc[interval_flags.pump_label.eq(label)&interval_flags.pump_interval_hybrid,'width_um']
            chosen=part.index.isin(flagged)&part.K.notna()
            if chosen.any():
                ax.scatter(part.index[chosen],part.loc[chosen,'K'],marker='D',color='darkorange',s=28,zorder=7)

    if table.any_hybrid.any():ax.scatter([],[],marker='D',color='crimson',s=65,label='Hybrid >10% в центре')
    if interval_flags is not None and interval_flags.pump_interval_hybrid.any():
        ax.scatter([],[],marker='D',color='darkorange',s=28,
            label=f'Накачка Hybrid в {100*hybridization.fraction:g}% JSI')
    ax.set(title='Число Шмидта от ширины',xlabel='Ширина, мкм',ylabel='Число Шмидта K',xticks=widths)
    ax.set_title(title or 'Число Шмидта от ширины',fontsize=16,pad=12)
    ax.set_xlabel('Ширина, мкм',fontsize=15);ax.set_ylabel('Число Шмидта K',fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=.25);ax.legend(fontsize=12,ncol=2)
    if standalone:
        fig.savefig(output/'schmidt_number_vs_width.png',dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)


def plot_raw_spatial_labels(run_dir, inspection, classification):
    from .interpolation_plots import plot_raw
    for width in sorted(inspection.pairs.width_um.unique()):
        tables={};branches={}
        for band in ('PUMP','IS'):
            table=inspection.selection.loc[inspection.selection.width_um.eq(width)&inspection.selection.band.eq(band)] if not inspection.selection.empty else pd.DataFrame()
            if not table.empty:tables[band]=table
            branch=inspection.branches.loc[inspection.branches.width_um.eq(width)&inspection.branches.band.eq(band)].copy() if not inspection.branches.empty else pd.DataFrame()
            if not branch.empty:
                rows=classification.loc[classification.width_um.eq(width)&classification.band.eq(band)&classification.internal_mode.notna()]
                labels={int(row['internal_mode']):f"{row['label']} (solnum {row['solnum']})" for row in rows.to_dict('records')}
                branch['mode']=branch['mode'].map(lambda mode:labels.get(int(mode),f'не классифицирована: {mode}'))
                branches[band]=branch
        manifest=json.loads((Path(run_dir)/'config.json').read_text(encoding='utf-8-sig'))
        plot_raw(inspection.output_dir/f'w_{width:g}'/'neff_raw.png',width,tables,branches,
                 imag_limit=inspection.tracking['imag_limit'],anchors={'PUMP':manifest['pump_wavelength_um'],'IS':2*manifest['pump_wavelength_um']})
