"""Signal/idler marginal densities and FWHM from the full cached JSI matrix."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from .frequency import LIGHT_SPEED_M_PER_FS
from .spatial_workflow import label_key, spatial_saved_is_current
from .cached_spectra import cached_case


@dataclass(frozen=True)
class HalfMaximum:
    width: float
    left: float
    right: float
    status: str
    intervals: int


def peak_normalize(density):
    values=np.asarray(density,dtype=float)
    if not np.isfinite(values).all() or np.any(values<0) or values.max()<=0:
        raise ValueError('Спектр должен быть конечным, неотрицательным и ненулевым')
    return values/values.max()


def fwhm(x,density):
    """Half-maximum crossings of the connected main peak, with linear interpolation."""
    x=np.asarray(x,dtype=float);y=peak_normalize(density)
    if x.shape!=y.shape or x.ndim!=1 or len(x)<3 or not np.isfinite(x).all() or np.any(np.diff(x)<=0):
        raise ValueError('Для FWHM нужна возрастающая конечная ось и согласованный спектр')
    above=y>=.5
    intervals=int(above[0])+int(np.count_nonzero(np.diff(above.astype(int))==1))
    center=int(np.argmax(y));lo=hi=center
    while lo>0 and above[lo-1]:lo-=1
    while hi<len(y)-1 and above[hi+1]:hi+=1
    left_clipped=lo==0 and y[0]>.5
    right_clipped=hi==len(y)-1 and y[-1]>.5
    left=(np.nan if left_clipped else x[0] if lo==0 else
          x[lo-1]+(.5-y[lo-1])*(x[lo]-x[lo-1])/(y[lo]-y[lo-1]))
    right=(np.nan if right_clipped else x[-1] if hi==len(y)-1 else
           x[hi]+(.5-y[hi])*(x[hi+1]-x[hi])/(y[hi+1]-y[hi]))
    clipped=left_clipped or right_clipped
    status=('both_edges_clipped' if left_clipped and right_clipped else 'left_edge_clipped' if left_clipped
            else 'right_edge_clipped' if right_clipped else 'multiple_halfmax_intervals' if intervals>1 else 'ok')
    return HalfMaximum(float('nan') if clipped else float(right-left),float(left),float(right),status,intervals)


def marginal_densities(omega,jsi):
    """JSI[idler,signal]; rectangle sums on the original uniform omega grid."""
    omega=np.asarray(omega,dtype=float);jsi=np.asarray(jsi,dtype=float)
    if (omega.ndim!=1 or len(omega)<3 or jsi.shape!=(len(omega),len(omega))
            or not np.isfinite(omega).all() or np.any(omega<=0) or np.any(np.diff(omega)<=0)
            or not np.isfinite(jsi).all() or np.any(jsi<0)):
        raise ValueError('Ожидались положительная возрастающая ω и конечная JSI[idler,signal]')
    spacing=np.diff(omega)
    if not np.allclose(spacing,spacing[0],rtol=1e-8,atol=1e-14):
        raise ValueError('Суммирование рассчитано на равномерную сетку ω')
    signal=jsi.sum(axis=0)*spacing[0]
    idler=jsi.sum(axis=1)*spacing[0]
    # Vacuum wavelength in nm; use the same rounded c as the rest of the project.
    constant=2*np.pi*LIGHT_SPEED_M_PER_FS*1e9
    wavelength=constant/omega
    order=np.argsort(wavelength)
    wavelength=wavelength[order]
    jacobian=constant/wavelength**2  # |dω/dλ_nm|, rad/(fs·nm)
    return dict(omega=omega,signal_omega=signal,idler_omega=idler,
                wavelength_nm=wavelength,signal_wavelength=signal[order]*jacobian,
                idler_wavelength=idler[order]*jacobian)


def _width_text(value,unit):
    if not np.isfinite(value.width):return 'FWHM: вне окна'
    suffix=' (главн. пик)' if value.intervals>1 else ''
    return f'FWHM = {value.width:.3g} {unit}'+suffix


def plot_marginal_spectra(run_dir,spatial,output,*,points=1501,dpi=150,cache_dir=None,title=None):
    """Read the full cached IMatrix, not the thinned image; no spectrum solver runs."""
    run_dir,output=Path(run_dir),Path(output)
    if not spatial_saved_is_current(run_dir,spatial,points,allow_cached_subset=True,output_dir=Path(cache_dir).parent if cache_dir is not None else output):
        raise ValueError('Сначала выполни расчёт JSI/K в разделе 5 с текущими параметрами')
    if not np.isscalar(dpi) or not np.isfinite(dpi) or dpi<=0:raise ValueError('dpi должен быть положительным')
    manifest=json.loads((run_dir/'config.json').read_text(encoding='utf-8-sig'))
    degree=spatial.study.degrees[0]
    summary=pd.read_csv((Path(cache_dir).parent if cache_dir is not None else output)/'summary.csv',float_precision='round_trip')
    widths=list(spatial.study.widths);labels=sorted(spatial.study.pump_labels,key=label_key)
    output.mkdir(parents=True,exist_ok=True)
    cache_dir=Path(cache_dir) if cache_dir is not None else output/'cache'
    cases={};records=[]
    for width,mapping in spatial.study.pump_map.items():
        for label,block in mapping.items():
            path,pulse=cached_case(cache_dir,spatial,width,label,summary,manifest,points)
            if not path.is_file():raise FileNotFoundError('Нет полной JSI в кэше; выполни раздел 5 с RUN_STUDY=True: '+str(path))
            with np.load(path,allow_pickle=False) as saved:
                spectra=marginal_densities(saved['w_grid'],saved['IMatrix'])
            for coordinate,xkey in (('omega','omega'),('wavelength','wavelength_nm')):
                for role in ('signal','idler'):
                    result=fwhm(spectra[xkey],spectra[f'{role}_{coordinate}'])
                    spectra[f'{role}_{coordinate}_fwhm']=result
                    records.append(dict(width_um=width,pump_label=label,degree=degree,coordinate=coordinate,
                        role=role,spatial_label=spatial.signal_label if role=='signal' else spatial.idler_label,
                        fwhm=result.width,left_halfmax=result.left,right_halfmax=result.right,
                        unit='rad/fs' if coordinate=='omega' else 'nm',status=result.status,
                        halfmax_intervals=result.intervals,points=points))
            cases[width,label]=spectra
    figures=[]
    (output/'figures').mkdir(exist_ok=True)
    for coordinate,xkey,xlabel,unit in (
        ('omega','omega','ω, рад/фс','рад/фс'),
        ('wavelength','wavelength_nm','λ, нм','нм'),
    ):
        fig=plt.figure(figsize=(max(6.0,3.8*len(labels)+.7),3.9*len(widths)+.4),layout='constrained')
        try:
            rows=fig.subfigures(len(widths),1,squeeze=False,hspace=.06)
            for row,width in enumerate(widths):
                subfig=rows[row,0];subfig.set_facecolor('#f1f4f7' if row%2==0 else '#ffffff')
                subfig.suptitle(f'w = {width:g} мкм',fontsize=17,fontweight='bold')
                axes=subfig.subplots(1,len(labels),squeeze=False)[0]
                for ax,label in zip(axes,labels):
                    if (width,label) not in cases:ax.set_visible(False);continue
                    spectra=cases[width,label];x=spectra[xkey]
                    for role,mode,color,style in (('signal',spatial.signal_label,'#1767a4','-'),('idler',spatial.idler_label,'#d56319','--')):
                        result=spectra[f'{role}_{coordinate}_fwhm']
                        y=peak_normalize(spectra[f'{role}_{coordinate}'])
                        ax.plot(x,y,color=color,ls=style,lw=1.6,label=f'{role} {mode}\n'+_width_text(result,unit))
                        if np.isfinite(result.width):
                            ax.plot([result.left,result.right],[.5,.5],color=color,lw=1,alpha=.6)
                    metadata=spatial.modes.loc[spatial.modes.width_um.eq(width)&spatial.modes.pump_label.eq(label)].iloc[0]
                    hybrid=bool(metadata.pump_hybrid or metadata.signal_hybrid or metadata.idler_hybrid)
                    ax.set_title(label+(' · Hybrid' if hybrid else ''),fontsize=12,color='crimson' if hybrid else 'black')
                    ax.set(xlabel=xlabel,ylabel='Нормированный спектр',ylim=(0,1.08))
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                    ax.tick_params(labelsize=9);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
            fig.suptitle((title+'\n' if title else '')+'Спектры signal/idler по '+('ω' if coordinate=='omega' else 'λ с якобианом'),fontsize=16 if len(labels)>1 else 12)
            path=output/'figures'/f'marginals_{coordinate}.png'
            fig.savefig(path,dpi=dpi);figures.append(path)
        finally:plt.close(fig)
    pd.DataFrame(records).to_csv(output/'marginal_fwhm.csv',index=False,encoding='utf-8-sig')
    return tuple(figures)
