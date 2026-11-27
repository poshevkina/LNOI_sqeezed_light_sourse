"""Exact one-dimensional projections of the full SPDC grid."""

import numpy as np


def sum_frequency_distribution(omega, jsi):
    """Sum anti-diagonals: IMatrix[idler,signal], constant omega_i+omega_s.

    Probabilities use the original rectangle-sum quadrature (domega**2).
    Return bin centres and normalized probability mass, without interpolation.
    """
    omega=np.asarray(omega,dtype=float);jsi=np.asarray(jsi,dtype=float)
    if (omega.ndim!=1 or len(omega)<3 or jsi.shape!=(len(omega),len(omega))
            or not np.isfinite(omega).all() or np.any(omega<=0)
            or np.any(np.diff(omega)<=0) or not np.isfinite(jsi).all() or np.any(jsi<0)):
        raise ValueError('Нужны конечная JSI[idler,signal] и возрастающая положительная ω')
    step=np.diff(omega)[0]
    if not np.allclose(np.diff(omega),step,rtol=1e-8,atol=1e-14):
        raise ValueError('Для суммирования антидиагоналей нужна равномерная сетка ω')
    mass=np.zeros(2*len(omega)-1)
    for row,values in enumerate(jsi):
        mass[row:row+len(omega)]+=values
    total=mass.sum()
    if not np.isfinite(total) or total<=0:raise ValueError('JSI должна иметь ненулевой конечный интеграл')
    return 2*omega[0]+step*np.arange(len(mass)),mass/total


def saved_sum_frequency(saved):
    if "omega_sum" in saved and "sum_frequency_mass" in saved:
        return saved["omega_sum"], saved["sum_frequency_mass"]
    return sum_frequency_distribution(saved["w_grid"], saved["IMatrix"])
