"""Resolve an existing full JSI using the parameters in its summary row."""
from pathlib import Path
import numpy as np
from .sweep import PulseConfig, _cache_path


def cached_case(cache,spatial,width,label,summary,manifest,points):
    rows=summary.loc[summary.width_um.eq(width)&summary.pump_label.eq(label)]
    if len(rows)!=1:raise ValueError(f'Неоднозначный случай JSI: w={width:g}, {label}')
    row=rows.iloc[0]
    if int(row.N_POINTS)!=points:raise ValueError('Сетка JSI не соответствует запрошенной')
    duration=float(row.t_fs);dw=float(row.DW)
    if not np.isfinite([duration,dw]).all() or min(duration,dw)<=0:raise ValueError('Некорректные параметры сохранённого импульса')
    durations=[int(duration),duration] if duration.is_integer() else [duration]
    for value in durations:
        pulse=PulseConfig(value,dw,points)
        path=_cache_path(Path(cache),spatial.study,width,spatial.study.degrees[0],
            spatial.study.pump_map[width][label],pulse,manifest.get('length_m',.12),False)
        if path.is_file():return path,pulse
    raise FileNotFoundError(f'Нет полной JSI для t={duration:g} фс, DW=±{dw:g}, w={width:g}, {label}. Сначала выполни расчёт серии.')
