"""Figures for comparing polynomial fits, widths and pulse durations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np

from .polynomial import DispersionStudy, width_label
from .frequency import um_to_omega


def width_token(width):
    return "local" if width is None else f"{width:g}"


DERIVATIVE_LABELS = [r"$n_{\rm eff}$", r"$dn_{\rm eff}/d\omega$, фс/рад",
                     r"$d^2n_{\rm eff}/d\omega^2$, фс$^2$/рад$^2$"]

# Цвет физической PUMP-моды одинаков во всех дисперсионных фигурах.
PUMP_COLORS = {"мода 1": "#174a9b", "мода 2": "#bf5100",
               "мода 3": "#16723b", "мода 4": "#af1646"}
IS_COLORS = {0: "#7038a5", 1: "#007c83"}
VARIATION_STYLES = ("-", "--", "-.", ":", (0, (7, 2)),
                    (0, (3, 1, 1, 1)), (0, (1, 1)))

def mode_color(branch, mode):
    if branch == "PUMP":
        return PUMP_COLORS.get(mode, plt.get_cmap("tab10")((int(mode.split()[-1]) - 1) % 10))
    return IS_COLORS.get(mode, plt.get_cmap("tab10")(mode % 10))

def shaded_color(base, index, count):
    # Смесь с белым: первый случай светлее, последний темнее, оттенок моды сохраняется.
    strength = 0.4 + 0.6 * index / max(1, count - 1)
    return tuple(1 - strength * (1 - np.asarray(to_rgb(base))))

class StudyPlots:
    def __init__(self, study: DispersionStudy, figure_dir: Path | None = None,
                 *, jsi_columns: int = 3, max_plot_points: int | None = None, show: bool = True):
        self.study = study
        self.show = show
        self.figure_dir = figure_dir
        self.jsi_columns = jsi_columns
        self.max_plot_points = max_plot_points
        if jsi_columns < 1 or (max_plot_points is not None and max_plot_points < 2):
            raise ValueError("Нужно задать положительное число столбцов и хотя бы две точки")

    def show_figure(self, fig, name):
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        if self.figure_dir is not None:
            self.figure_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.figure_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)


    def is_mode_label(self, mode):
        roles = []
        if mode == self.study.signal_mode:
            roles.append("Signal")
        if mode == self.study.idler_mode:
            roles.append("Idler")
        return f"{' / '.join(roles) or 'IS'} (блок {mode})"

    def dispersion_figure(self, row_entries, variations, title, name):
        # row_entries[branch]: (физическая PUMP-метка / IS-номер, вариант, данные, полином)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
        for row, branch in enumerate(("PUMP", "IS")):
            seen_points, modes = set(), []
            for mode, variant, data, poly in row_entries[branch]:
                if mode not in modes:
                    modes.append(mode)
                omega, neff = um_to_omega(data.wavelength_um), data.neff
                dense = np.linspace(omega.min(), omega.max(), 700)
                color = shaded_color(mode_color(branch, mode), variant, len(variations))
                style = VARIATION_STYLES[variant % len(VARIATION_STYLES)]
                for order in (0, 1, 2):
                    axes[row, order].plot(dense, poly.deriv(order)(dense),
                                          color=color, ls=style, lw=1.6)
                # Точки одной моды не дублируются для разных степеней одного набора данных.
                point_key = (mode, id(data))
                if point_key not in seen_points:
                    axes[row, 0].scatter(omega, neff, s=8, color=mode_color(branch, mode), alpha=0.45)
                    seen_points.add(point_key)
            handles = [Line2D([0], [0], color=mode_color(branch, mode), lw=2,
                              label=f"PUMP {mode}" if branch == "PUMP" else self.is_mode_label(mode))
                       for mode in modes]
            axes[row, 0].legend(handles=handles, fontsize=8, loc="best")
            center = self.study.pump_wavelength_um if branch == "PUMP" else 2 * self.study.pump_wavelength_um
            center = float(um_to_omega(center))
            for order, ax in enumerate(axes[row]):
                ax.set(xlabel=r"$\omega$, рад/фс", ylabel=DERIVATIVE_LABELS[order],
                       title=f"{branch}: " + ("дисперсия", "первая производная", "вторая производная")[order])
                ax.axvline(center, color="black", ls="--", lw=0.8, alpha=0.4)
                ax.grid(alpha=0.25)
        variant_handles = [Line2D([0], [0], color=shaded_color("#111111", i, len(variations)),
                                  ls=VARIATION_STYLES[i % len(VARIATION_STYLES)], lw=2, label=label)
                           for i, label in enumerate(variations)]
        fig.legend(handles=variant_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=min(7, len(variations)), fontsize=9,
                   title="Оттенок и стиль: от светлого к тёмному")
        fig.suptitle(title)
        self.show_figure(fig, name)

    def plot_dispersion_by_width(self):
        for w in self.study.widths:
            entries = {"PUMP": [], "IS": []}
            for branch in entries:
                mapping = self.study.pump_map[w] if branch == "PUMP" else {m: m for m in self.study.raw[w]["IS"]}
                for label, mode in mapping.items():
                    for i, degree in enumerate(self.study.degrees):
                        entries[branch].append((label, i, self.study.raw[w][branch][mode],
                                                 self.study.fits[w][degree][branch][mode]))
            self.dispersion_figure(entries, [f"степень {d}" for d in self.study.degrees],
                              f"{width_label(w)} | все моды, сравнение полиномов", f"disp_width_{width_token(w)}")

    def plot_dispersion_by_degree(self):
        for degree in self.study.degrees:
            entries = {"PUMP": [], "IS": []}
            for i, w in enumerate(self.study.widths):
                for branch in entries:
                    mapping = self.study.pump_map[w] if branch == "PUMP" else {m: m for m in self.study.raw[w]["IS"]}
                    for label, mode in mapping.items():
                        entries[branch].append((label, i, self.study.raw[w][branch][mode],
                                                 self.study.fits[w][degree][branch][mode]))
            self.dispersion_figure(entries, [f"{width_label(w)}" for w in self.study.widths],
                              f"Степень {degree} | все моды, сравнение ширин", f"disp_degree_{degree}")

    def jsi_panel(self, ax, result):
        with np.load(result["cache_path"], allow_pickle=False) as saved:
            grid = saved["lmb_grid"]
            count = len(grid) if self.max_plot_points is None else min(self.max_plot_points, len(grid))
            indices = np.unique(np.linspace(0, len(grid) - 1, count, dtype=int))
            image = saved["IMatrix"][np.ix_(indices, indices)]
        mesh = ax.pcolormesh(grid[indices], grid[indices], image,
                             shading="auto", cmap="inferno", vmin=0, vmax=1, rasterized=True)
        ax.set(xlabel=r"$\lambda_s$, нм", ylabel=r"$\lambda_i$, нм")
        return mesh

    def plot_jsi_by_width(self, results, config):
        for w in self.study.widths:
            labels = list(self.study.pump_map[w])
            fig, axes = plt.subplots(len(labels), len(self.study.degrees),
                                     figsize=(4.5 * len(self.study.degrees), 3.8 * len(labels)), squeeze=False)
            for row, label in enumerate(labels):
                for col, degree in enumerate(self.study.degrees):
                    r = results[w][degree][label]
                    mesh = self.jsi_panel(axes[row, col], r)
                    axes[row, col].set_title(f"d={degree} | PUMP {label}\n"
                                             f"K={r['K']:.4f}; Λ={r['Lambda_um']:.3f} мкм", fontsize=9)
                    fig.colorbar(mesh, ax=axes[row, col], label="JSI")
            fig.suptitle(f"t={config.t_fs:g} фс | {width_label(w)} | сравнение полиномов")
            self.show_figure(fig, f"{config.tag}_JSI_width_{width_token(w)}")

    def plot_jsi_by_degree(self, results, config):
        for degree in self.study.degrees:
            for offset in range(0, len(self.study.widths), self.jsi_columns):
                subset = self.study.widths[offset:offset + self.jsi_columns]
                labels = [label for label in self.study.pump_labels if any(label in self.study.pump_map[w] for w in subset)]
                fig, axes = plt.subplots(len(labels), len(subset),
                                         figsize=(4.5 * len(subset), 3.8 * len(labels)), squeeze=False)
                for row, label in enumerate(labels):
                    for col, w in enumerate(subset):
                        ax = axes[row, col]
                        if label not in results[w][degree]:
                            ax.text(0.5, 0.5, f"{width_label(w)}\nPUMP {label}\nнет в выбранных данных",
                                    ha="center", va="center", transform=ax.transAxes)
                            ax.set_axis_off()
                            continue
                        r = results[w][degree][label]
                        mesh = self.jsi_panel(ax, r)
                        ax.set_title(f"{width_label(w)} | PUMP {label}\n"
                                     f"K={r['K']:.4f}; Λ={r['Lambda_um']:.3f} мкм", fontsize=9)
                        fig.colorbar(mesh, ax=ax, label="JSI")
                fig.suptitle(f"t={config.t_fs:g} фс | степень {degree} | сравнение ширин")
                self.show_figure(fig, f"{config.tag}_JSI_degree_{degree}_page_{offset // self.jsi_columns + 1}")

    def spectra_panel(self, ax, entries):
        for label, r in entries:
            line, = ax.plot(r["lmb_grid"], r["S_s_norm"], lw=1.6,
                            label=f"{label}, s; K={r['K']:.3f}")
            ax.plot(r["lmb_grid"], r["S_i_norm"], ls="--", lw=1.6,
                    color=line.get_color(), label=f"{label}, i")
        ax.set(xlabel=r"$\lambda$, нм", ylabel="Нормированный спектр", ylim=(0, 1.05))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    def plot_spectra_by_width(self, results, config):
        for w in self.study.widths:
            labels = list(self.study.pump_map[w])
            fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4.8), squeeze=False)
            for ax, label in zip(axes.ravel(), labels):
                self.spectra_panel(ax, [(f"d={d}", results[w][d][label]) for d in self.study.degrees])
                ax.set_title(f"PUMP {label}")
            fig.suptitle(f"t={config.t_fs:g} фс | {width_label(w)} | спектры: s — сплошная, i — штриховая")
            self.show_figure(fig, f"{config.tag}_spectra_width_{width_token(w)}")

    def plot_spectra_by_degree(self, results, config):
        for degree in self.study.degrees:
            for label_number, label in enumerate(self.study.pump_labels):
                entries = [(f"{width_label(w)}", results[w][degree][label]) for w in self.study.widths
                           if label in results[w][degree]]
                if not entries:
                    continue
                fig, ax = plt.subplots(figsize=(8.5, 5.2))
                self.spectra_panel(ax, entries)
                fig.suptitle(f"t={config.t_fs:g} фс | d={degree} | PUMP {label}\n"
                             "Сравнение ширин; s — сплошная, i — штриховая")
                self.show_figure(fig, f"{config.tag}_spectra_degree_{degree}_label_{label_number}")

    def plot_summary(self, summary, config):
        if len(self.study.widths) < 2:
            return
        for number, label in enumerate(self.study.pump_labels):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for degree in self.study.degrees:
                part = summary[(summary.degree == degree) & (summary.pump_label == label)]
                if part.empty:
                    continue
                part = part.set_index("width_um").reindex(self.study.widths)
                # Пропущенная мода оставляет разрыв, а не соединяет соседние ширины.
                axes[0].plot(part.index, part.K, "o-", label=f"d={degree}")
                axes[1].plot(part.index, part.Lambda_um, "o-", label=f"d={degree}")
            axes[0].set(ylabel="Число Шмидта K")
            axes[1].set(ylabel="Период полинга |Λ|, мкм")
            for ax in axes:
                ax.set_xlabel("Ширина w, мкм")
                ax.grid(alpha=0.25)
                ax.legend()
            fig.suptitle(f"t={config.t_fs:g} фс | PUMP {label}")
            self.show_figure(fig, f"{config.tag}_summary_label_{number}")
