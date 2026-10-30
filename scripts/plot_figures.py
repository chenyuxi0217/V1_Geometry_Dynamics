import os
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. Parameters, paths, and global style
# ============================================================

np.random.seed(42)
warnings.filterwarnings("ignore", module=r"^allensdk(?:\.|$)")
warnings.filterwarnings("ignore", module=r"^hdmf(?:\.|$)")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURE_DIR = os.path.join(DATA_DIR, "figures")
FIGURE_DPI = 300
LABEL_SIZE = 16
TICK_SIZE = 14.5
LEGEND_SIZE = 14
NOTE_SIZE = 14
PANEL_SIZE = 21
PANEL_X_PAD_POINTS = 36
PANEL_Y_PAD_POINTS = 18
TABLE_SIZE = 13.5
os.makedirs(FIGURE_DIR, exist_ok=True)
BIN_MS_LIST = [20, 50, 100]
TOP_MODES = 20
SUBSAMPLING_SHARED_TAU_AXIS = False
NEURON_COUNT_VALUES = [40, 50, 60]
NEURON_COUNT_MARKERS = {40: "o", 50: "s", 60: "^"}
NEURON_COUNT_CORR_METRICS = [
    ("mean", r"Mean $\tau$"),
    ("log_spectrum_slope", r"Slope $b$")
]
COLOR_G = "#2A9D8F"
COLOR_M = "#E76F51"
COLOR_DELTA = "#264653"
BIN_COLORS = {20: "#3A86FF", 50: "#8338EC", 100: "#FF7F0E"}
CORR_METRICS = [
    ("mean", r"Mean $\tau$" + "\n(ranks 1–20)"),
    ("slow3_mean", "Slow-tail mean\n(ranks 1–3)"),
    ("slow10_mean", "Slower-half mean\n(ranks 1–10)"),
    ("lower10_mean", "Faster-half mean\n(ranks 11–20)"),
    ("log_spectrum_slope", r"Log-spectrum slope $b$" + "\n(ranks 1–20)")
]
LOG_RATIO_CORR_METRICS = [
    ("slow_lower_logratio", r"Tail log-ratio $C_{\mathrm{tail}}$"),
    ("half_logratio", r"Half log-ratio $C_{\mathrm{half}}$")
]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": LABEL_SIZE,
    "axes.titlesize": LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    "axes.linewidth": 0.8,
    "axes.labelpad": 7,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "savefig.dpi": FIGURE_DPI
})


# ============================================================
# 2. Functions
# ============================================================

# ------------------------------------------------------------
# CSV reading
# ------------------------------------------------------------


def read_csv(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))


# ------------------------------------------------------------
# Figure saving, labels, formatting, and basic summaries
# ------------------------------------------------------------


def save_figure(fig, stem):
    png_path = os.path.join(FIGURE_DIR, f"{stem}.png")
    tif_path = os.path.join(FIGURE_DIR, f"{stem}.tiff")
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(tif_path, dpi=FIGURE_DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def add_panel_labels(fig, panels, figure_style=None, panel_styles=None):
    """
    x_pad_points, y_pad_points, dx_points, dy_points are in points (1/72 inch).
    x and y, if provided, are in figure fractions and override the automatically computed position.

    Example: figure_style={"y_pad_points": 14},
             panel_styles={"B": {"dx_points": 4, "dy_points": -3}}.
    """
    defaults = {
        "x_pad_points": PANEL_X_PAD_POINTS, "y_pad_points": PANEL_Y_PAD_POINTS,
        "dx_points": 0.0, "dy_points": 0.0, "x": None, "y": None,
        "fontsize": PANEL_SIZE, "fontweight": "bold", "color": plt.rcParams["text.color"]
    }
    figure_style = {} if figure_style is None else figure_style
    panel_styles = {} if panel_styles is None else panel_styles
    if not isinstance(panels, dict) or not isinstance(figure_style, dict) or not isinstance(panel_styles, dict):
        raise TypeError("panels, figure_style, and panel_styles must be dictionaries.")
    unknown = set(figure_style) - set(defaults)
    if unknown: raise ValueError(f"Unknown figure label settings: {sorted(unknown)}")
    unknown_panels = set(panel_styles) - set(panels)
    if unknown_panels: raise ValueError(f"Styles refer to absent panels: {sorted(unknown_panels)}")
    base = {**defaults, **figure_style}
    prepared = []
    for letter, group in panels.items():
        override = panel_styles.get(letter, {})
        if not isinstance(override, dict): raise TypeError(f"Panel {letter}: style must be a dictionary.")
        unknown = set(override) - set(defaults)
        if unknown: raise ValueError(f"Panel {letter}: unknown label settings {sorted(unknown)}")
        style = {**base, **override}
        for name in ("x_pad_points", "y_pad_points", "dx_points", "dy_points", "fontsize", "x", "y"):
            if name in ("x", "y") and style[name] is None: continue
            value = float(style[name])
            if not np.isfinite(value): raise ValueError(f"Panel {letter}: {name} must be finite.")
            style[name] = value
        if style["fontsize"] <= 0: raise ValueError(f"Panel {letter}: fontsize must be positive.")
        axes = [group] if hasattr(group, "get_position") else list(np.asarray(group, dtype=object).ravel())
        if not axes or any(getattr(ax, "figure", None) is not fig for ax in axes):
            raise ValueError(f"Panel {letter}: expected nonempty axes from the same figure.")
        prepared.append((letter, axes, style))

    for artist in list(fig.texts):
        if artist.get_gid() == "panel_label": artist.remove()
    fig.canvas.draw()
    width_points, height_points = 72.0 * fig.get_figwidth(), 72.0 * fig.get_figheight()
    labels = []
    for letter, axes, style in prepared:
        boxes = [ax.get_position(original=True) for ax in axes]
        x = min(box.x0 for box in boxes) - style["x_pad_points"] / width_points
        y = max(box.y1 for box in boxes) + style["y_pad_points"] / height_points
        if style["x"] is not None: x = style["x"]
        if style["y"] is not None: y = style["y"]
        x += style["dx_points"] / width_points
        y += style["dy_points"] / height_points
        labels.append(fig.text(x, y, letter, transform=fig.transFigure, fontsize=style["fontsize"],
                               fontweight=style["fontweight"], color=style["color"], ha="left",
                               va="bottom", clip_on=False, gid="panel_label"))
    return labels


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def format_p(p):
    if not np.isfinite(p): return "p=n/a"
    if p == 0: return "p<1e-300"
    return f"p={p:.3g}"


def p_stars(q):
    if not np.isfinite(q): return ""
    if q < 0.001: return "***"
    if q < 0.01: return "**"
    if q < 0.05: return "*"
    return "ns"


def summary_row(bin_ms, metric):
    sub = SUMMARY_DF[(SUMMARY_DF["bin_ms"] == bin_ms) & (SUMMARY_DF["Metric"] == metric)]
    if len(sub) != 1: raise ValueError(f"Expected one summary row for bin={bin_ms}, metric={metric}; got {len(sub)}")
    return sub.iloc[0]


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


# ------------------------------------------------------------
# Main condition effects, geometry, and temporal profiles
# ------------------------------------------------------------


def plot_condition_metric_across_bins(ax, metric, ylabel, *, annotate=True):
    rows = [summary_row(b, metric) for b in BIN_MS_LIST]
    g = np.array([r["Gratings_mean"] for r in rows], dtype=float)
    m = np.array([r["Movie_mean"] for r in rows], dtype=float)
    g_sem = np.array([r["Gratings_sem"] for r in rows], dtype=float)
    m_sem = np.array([r["Movie_sem"] for r in rows], dtype=float)

    ax.errorbar(BIN_MS_LIST, g, yerr=g_sem, marker="o", ms=5, lw=2, capsize=3, color=COLOR_G, label="Gratings")
    ax.errorbar(BIN_MS_LIST, m, yerr=m_sem, marker="o", ms=5, lw=2, capsize=3, color=COLOR_M, label="Movie")
    ax.set_xticks(BIN_MS_LIST)
    ax.set_xlabel("Bin width (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    despine(ax)

    if annotate:
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        for b, r, gv, mv in zip(BIN_MS_LIST, rows, g, m):
            y = max(gv + float(r["Gratings_sem"]), mv + float(r["Movie_sem"])) + 0.04 * span
            ax.text(b, y, p_stars(float(r["q_bh_family"])), ha="center", va="bottom", fontsize=NOTE_SIZE)
        ax.set_ylim(y_min, y_max + 0.10 * span)


def plot_condition_metric_at_bin(ax, summary, bin_ms, metric, ylabel, *, show_n=False):
    selected = summary[(summary["bin_ms"] == bin_ms) & (summary["Metric"] == metric)]
    if len(selected) != 1:
        raise ValueError(f"Expected one paired summary row for {bin_ms} ms, {metric}; got {len(selected)}.")
    row = selected.iloc[0]
    means = np.array([row["Gratings_mean"], row["Movie_mean"]], dtype=float)
    errors = np.array([row["Gratings_sem"], row["Movie_sem"]], dtype=float)
    for x, mean, error, color, label in zip([0, 1], means, errors, [COLOR_G, COLOR_M], ["Gratings", "Movie"]):
        if np.isfinite(mean):
            ax.errorbar(x, mean, yerr=error if np.isfinite(error) else None, fmt="o", ms=6,
                        color=color, capsize=4, lw=1.8, label=label, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Gratings", "Movie"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.locator_params(axis="y", nbins=4)
    finite = np.isfinite(means)
    if finite.any():
        spread = np.where(np.isfinite(errors), errors, 0.0)
        bottom = float(np.min(means[finite] - spread[finite]))
        top = float(np.max(means[finite] + spread[finite]))
        span = max(top - bottom, 0.05 * max(float(np.max(np.abs(means[finite]))), 1e-3), 1e-6)
        ax.set_ylim(bottom - 0.18 * span, top + 0.43 * span)
        if finite.all():
            stars = p_stars(float(row["q_bh_family"]))
            if stars:
                y = top + 0.13 * span
                ax.plot([0, 0, 1, 1], [y, y + 0.06 * span, y + 0.06 * span, y],
                        color="0.35", lw=0.8)
                ax.text(0.5, top + 0.22 * span, stars, ha="center", va="bottom", fontsize=NOTE_SIZE)
    if not finite.all():
        ax.text(0.5, 0.5, "No complete paired estimates", ha="center", va="center",
                transform=ax.transAxes, fontsize=NOTE_SIZE)
    if show_n:
        n = row["n_pairs"]
        n_text = str(int(n)) if np.isfinite(n) else "n/a"
        ax.text(0.98, -0.32, f"Paired n={n_text}", ha="right", va="top",
                transform=ax.transAxes, fontsize=NOTE_SIZE, clip_on=False)
    despine(ax)


def plot_paired_condition_metric(ax, bin_ms, col_g, col_m, ylabel, color_g=COLOR_G, color_m=COLOR_M):
    df = PER_SESSION_DF[PER_SESSION_DF["bin_ms"] == bin_ms][[col_g, col_m]].dropna()
    g = df[col_g].to_numpy(dtype=float)
    m = df[col_m].to_numpy(dtype=float)

    for gv, mv in zip(g, m):
        ax.plot([0, 1], [gv, mv], color="0.75", lw=0.8, alpha=0.65, zorder=1)

    ax.scatter(np.zeros_like(g), g, s=24, color=color_g, alpha=0.80, edgecolor="white", linewidth=0.4, zorder=2)
    ax.scatter(np.ones_like(m), m, s=24, color=color_m, alpha=0.80, edgecolor="white", linewidth=0.4, zorder=2)

    means = [np.mean(g), np.mean(m)]
    errors = [sem(g), sem(m)]
    ax.errorbar([0, 1], means, yerr=errors, fmt="o", ms=7, color="black", capsize=4, lw=1.5, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Gratings", "Movie"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    despine(ax)


def plot_pca_panel(ax, bin_ms=100, use_std=True):
    df = PCA_SUMMARY_DF[PCA_SUMMARY_DF["bin_ms"] == bin_ms].sort_values("PC")
    spread = "std" if use_std else "sem"
    x = df["PC"].to_numpy()
    g = df["Grating_mean"].to_numpy()
    m = df["Movie_mean"].to_numpy()
    gs = df[f"Grating_{spread}"].to_numpy()
    ms = df[f"Movie_{spread}"].to_numpy()

    ax.fill_between(x, g - gs, g + gs, color=COLOR_G, alpha=0.18, edgecolor="none")
    ax.fill_between(x, m - ms, m + ms, color=COLOR_M, alpha=0.18, edgecolor="none")
    ax.plot(x, g, color=COLOR_G, lw=2, label="Gratings")
    ax.plot(x, m, color=COLOR_M, lw=2, label="Movie")
    ax.set_xlim(1, int(df["PC"].max()))
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xlabel("Principal component rank")
    ax.set_ylabel("Cumulative variance")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    despine(ax)


def plot_acf_panel(ax, bin_ms, show_legend=False):
    df = ACF_DF[ACF_DF["bin_ms"] == bin_ms]
    for col, label, color in [("g_acf_norm", "Gratings", COLOR_G), ("m_acf_norm", "Movie", COLOR_M)]:
        grp = df.groupby("lag_ms")[col].agg(["mean", "std", "count"]).reset_index()
        grp["sem"] = grp["std"] / np.sqrt(grp["count"])
        x = grp["lag_ms"].to_numpy()
        y = grp["mean"].to_numpy()
        e = grp["sem"].to_numpy()
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, edgecolor="none")
        ax.plot(x, y, color=color, lw=2, label=label)

    ax.axhline(0, color="0.4", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized ACF")
    if show_legend: ax.legend(frameon=False)
    despine(ax)


def plot_tau_spectrum_panel(ax, bin_ms, show_legend=False):
    df = TAU_SPECTRUM_DF[TAU_SPECTRUM_DF["bin_ms"] == bin_ms]
    for col, label, color in [("g_tau", "Gratings", COLOR_G), ("m_tau", "Movie", COLOR_M)]:
        grp = df.groupby("rank")[col].agg(["mean", "std", "count"]).reset_index()
        grp["sem"] = grp["std"] / np.sqrt(grp["count"])
        x = grp["rank"].to_numpy()
        y = grp["mean"].to_numpy()
        e = grp["sem"].to_numpy()
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, edgecolor="none")
        ax.plot(x, y, marker="o", ms=3.5, color=color, lw=1.8, label=label)

    ax.set_xlim(1, TOP_MODES)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Retained rank (1 = slowest)")
    ax.set_ylabel(r"$\tau$ (ms)")
    if show_legend: ax.legend(frameon=False)
    despine(ax)


# ------------------------------------------------------------
# Combined correlations and spectral-shape plots
# ------------------------------------------------------------


def build_combined_correlation_table():
    scalar = TAU_SUMMARY_CORR_DF.copy()
    scalar["source"] = "summary"
    shape = TAU_SHAPE_CORR_DF.copy()
    shape["source"] = "shape"
    return pd.concat([scalar, shape], ignore_index=True, sort=False)


def plot_correlation_forest(ax, condition, metrics=None):
    metrics = CORR_METRICS if metrics is None else metrics
    y_base = np.arange(len(metrics))[::-1]
    offsets = {20: 0.20, 50: 0.0, 100: -0.20}

    ax.axvline(0, color="0.45", lw=1, ls="--", zorder=0)
    for b in BIN_MS_LIST:
        for i, (metric, _) in enumerate(metrics):
            sub = CORR_DF[(CORR_DF["condition"] == condition) & (CORR_DF["bin_ms"] == b) & (CORR_DF["metric"] == metric)]
            if len(sub) != 1: raise ValueError(f"Expected one correlation row for {condition}, {b} ms, {metric}; got {len(sub)}.")
            r = sub.iloc[0]
            y = y_base[i] + offsets[b]
            rho, lo, hi = float(r["rho"]), float(r["ci95_low"]), float(r["ci95_high"])
            draw_horizontal_ci(ax, rho, lo, hi, y, BIN_COLORS[b], marker="s", label=f"{b} ms" if i == 0 else None)

    ax.set_yticks(y_base)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"Partial Spearman $\rho$")
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    despine(ax)


def plot_mode_corr(ax, condition, show_ci=True, legend=True):
    for b in BIN_MS_LIST:
        df = TAU_MODE_CORR_DF[(TAU_MODE_CORR_DF["condition"] == condition) & (TAU_MODE_CORR_DF["bin_ms"] == b)].sort_values("rank")
        x = df["rank"].to_numpy()
        y = df["rho"].to_numpy()
        ax.plot(x, y, marker="o", ms=3.5, lw=1.8, color=BIN_COLORS[b], label=f"{b} ms")
        if show_ci:
            ax.fill_between(x, df["ci95_low"].to_numpy(), df["ci95_high"].to_numpy(), color=BIN_COLORS[b], alpha=0.10, edgecolor="none")

    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlim(1, TOP_MODES)
    ax.set_ylim(-1, 1)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Retained rank (1 = slowest)")
    ax.set_ylabel(r"Partial Spearman $\rho$" + "\n" + r"$(D_{eff},\tau_{rank})$")
    if legend: ax.legend(frameon=False, ncol=3)
    despine(ax)


def get_shape_corr(bin_ms, condition="Movie", metric="slow_lower_logratio"):
    sub = TAU_SHAPE_CORR_DF[(TAU_SHAPE_CORR_DF["bin_ms"] == bin_ms) & (TAU_SHAPE_CORR_DF["condition"] == condition) & (TAU_SHAPE_CORR_DF["metric"] == metric)]
    if len(sub) != 1: raise ValueError(f"Missing shape correlation for {bin_ms}, {condition}, {metric}")
    return sub.iloc[0]


def plot_movie_shape_scatter(ax, bin_ms, metric, value_column, ylabel, stats_ax=None):
    d1 = PER_SESSION_DF.loc[PER_SESSION_DF["bin_ms"] == bin_ms, ["session_id", "m_exp_dim", "movie_speed_mean"]]
    d2 = TAU_SHAPE_SESSION_DF.loc[TAU_SHAPE_SESSION_DF["bin_ms"] == bin_ms, ["session_id", value_column]]
    df = d1.merge(d2, on="session_id", how="inner", validate="one_to_one").replace([np.inf, -np.inf], np.nan).dropna()
    row = get_shape_corr(bin_ms, condition="Movie", metric=metric)
    ax.scatter(df["m_exp_dim"], df[value_column], s=48, color=COLOR_M, alpha=0.88, edgecolor="white", linewidth=0.6)
    if not len(df): ax.text(0.5, 0.5, "No valid paired session values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"Movie $D_{eff}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20)
    ax.margins(x=0.08, y=0.10)
    if np.isfinite(row["rho"]):
        n = int(row.get("n_sessions", len(df)))
        text = f"$\\rho_{{partial}}$={row['rho']:.2f}; n={n}"
        text += f"\n95% pointwise CI [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]"
        text += f"\n{format_p(float(row['p']))} (unadjusted)"
        text += f"\n$q_{{BH}}$={row['q_bh_family']:.3g}"
        text += f"\n$q_{{BY}}$={row['q_by_family']:.3g}"
    else:
        text = f"Correlation not estimable\nn={int(row.get('n_sessions', len(df)))}"
    if stats_ax is None:
        ax.text(0.02, -0.40, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=NOTE_SIZE, clip_on=False, linespacing=1.2)
    else:
        stats_ax.axis("off")
        stats_ax.text(0, 0.95, text, transform=stats_ax.transAxes, ha="left", va="top",
                      fontsize=NOTE_SIZE, linespacing=1.2, clip_on=False)
    despine(ax)


def plot_movie_hierarchy_scatter(ax, bin_ms, stats_ax=None):
    plot_movie_shape_scatter(ax, bin_ms, "slow_lower_logratio", "m_tau_slow_lower_logratio", r"Tail log-ratio $C_{\mathrm{tail}}$", stats_ax=stats_ax)


def plot_movie_half_ratio_scatter(ax, bin_ms, stats_ax=None):
    plot_movie_shape_scatter(ax, bin_ms, "half_logratio", "m_tau_half_logratio", r"Half log-ratio $C_{\mathrm{half}}$", stats_ax=stats_ax)


def plot_movie_slope_scatter(ax, bin_ms, stats_ax=None):
    plot_movie_shape_scatter(ax, bin_ms, "log_spectrum_slope", "m_tau_log_spectrum_slope", r"Log-spectrum slope $b$", stats_ax=stats_ax)


def draw_horizontal_ci(ax, estimate, lo, hi, y, color, marker="o", label=None, cap=0.06, alpha=1.0):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.hlines(y, lo, hi, color=color, lw=1.3, alpha=alpha)
        ax.vlines([lo, hi], y - cap, y + cap, color=color, lw=1.0, alpha=alpha)
    if np.isfinite(estimate):
        ax.plot(estimate, y, marker=marker, ms=5, color=color, linestyle="none", alpha=alpha, label=label, markerfacecolor="none" if marker == "o" else color)


# ------------------------------------------------------------
# Within-session sampling and repeat-average plots
# ------------------------------------------------------------
def plot_subsampling_distribution(ax, raw, order, value_col):
    rng = np.random.RandomState(42)
    groups = [raw.loc[raw["session_id"] == sid, value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for sid in order]
    positions = [j for j, values in enumerate(groups) if len(values)]
    values = [groups[j] for j in positions]
    if values:
        ax.boxplot(values, positions=positions, widths=0.55, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "lightgray", "linewidth": 0.8}, medianprops={"color": "0.25"})
        for pos, group in zip(positions, values):
            ax.scatter(pos + rng.uniform(-0.22, 0.22, len(group)), group, color=COLOR_DELTA, alpha=0.45, s=5, edgecolors="none")
    if len(order): ax.set_xlim(-0.7, len(order) - 0.3)


def plot_psth_condition_metric(ax, bin_ms, metric, ylabel, show_n=True):
    plot_condition_metric_at_bin(ax, PSTH_SUMMARY_DF, bin_ms, metric, ylabel, show_n=show_n)


def plot_raw_psth_correlation_comparison(ax, condition, show_labels=True, show_legend=True, show_n=True):
    specs = [(metric, b) for metric in ("mean", "log_spectrum_slope") for b in BIN_MS_LIST]
    ax.axvline(0, ls="--", lw=0.9, color="0.45")
    for j, (metric, b) in enumerate(specs):
        base_y = len(specs) - 1 - j
        for representation, marker, offset in (("Raw", "o", 0.14), ("PSTH", "s", -0.14)):
            d = RAW_PSTH_CORR_DF[(RAW_PSTH_CORR_DF["condition"] == condition) & (RAW_PSTH_CORR_DF["metric"] == metric)
                                 & (RAW_PSTH_CORR_DF["bin_ms"] == b) & (RAW_PSTH_CORR_DF["representation"] == representation)]
            if len(d) != 1: raise ValueError(f"Missing matched Raw/PSTH comparison: {condition}, {metric}, {b}, {representation}")
            r = d.iloc[0]
            draw_horizontal_ci(ax, float(r["rho"]), float(r["ci95_low"]), float(r["ci95_high"]), base_y + offset, BIN_COLORS[b], marker=marker)
        n = int(d.iloc[0]["n_sessions"])
        if show_n: ax.text(1.01, base_y, f"n={n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=NOTE_SIZE)
    ax.set_yticks(np.arange(len(specs))[::-1])
    ax.set_yticklabels([("Mean $\\tau$" if metric == "mean" else "Slope $b$") + f", {b} ms" for metric, b in specs], fontsize=NOTE_SIZE)
    if not show_labels: ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.55, len(specs) - 0.45)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"Partial Spearman $\rho$")
    ax.axhline(2.5, color="0.85", lw=0.8)
    handles = [Line2D([], [], marker="o", color="0.35", markerfacecolor="none", linestyle="none", label="Unaveraged"),
               Line2D([], [], marker="s", color="0.35", linestyle="none", label="PSTH")]
    if show_legend: ax.legend(handles=handles, frameon=False, fontsize=LEGEND_SIZE, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
    despine(ax)


def plot_psth_qc_by_condition(ax, field, ylabel, log10=False, bounds=None):
    for stim, color, label in (("g", COLOR_G, "Gratings"), ("m", COLOR_M, "Movie")):
        means, errors = [], []
        for b in BIN_MS_LIST:
            v = pd.to_numeric(PSTH_PER_DF.loc[PSTH_PER_DF["bin_ms"] == b, f"{stim}_{field}"], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if log10: v = np.log10(v[v > 0])
            means.append(float(np.mean(v)) if len(v) else np.nan)
            errors.append(sem(v))
        if np.isfinite(means).any(): ax.errorbar(BIN_MS_LIST, means, yerr=errors, marker="o", lw=1.8, capsize=3, color=color, label=label)
    ax.set_xticks(BIN_MS_LIST)
    ax.set_xlabel("Bin width (ms)")
    ax.set_ylabel(ylabel)
    if bounds is not None: ax.set_ylim(*bounds)
    ax.legend(frameon=False, fontsize=NOTE_SIZE)
    despine(ax)


def plot_ols_prediction_qc(ax, frame, ylabel):
    for stim, color, label in (("g", COLOR_G, "Gratings"), ("m", COLOR_M, "Movie")):
        for field, style, suffix in (("exp_ols_test_r2", "-", "real"), ("shuff_ols_test_r2", "--", "shuffle")):
            means, errors = [], []
            for b in BIN_MS_LIST:
                values = pd.to_numeric(frame.loc[frame["bin_ms"] == b, f"{stim}_{field}"], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                means.append(float(values.mean()) if len(values) else np.nan)
                errors.append(sem(values))
            if np.isfinite(means).any():
                marker = "o" if stim == "g" else "s"
                face = color if suffix == "real" else "none"
                ax.errorbar(BIN_MS_LIST, means, yerr=errors, marker=marker, mfc=face, ms=6, color=color,
                            ls=style, lw=1.8, capsize=3, alpha=1.0 if suffix == "real" else 0.7, label=f"{label}: {suffix}")
    ax.axhline(0, color="0.5", ls=":", lw=0.9)
    ax.set_xticks(BIN_MS_LIST)
    ax.set_xlabel("Bin width (ms)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE - 1, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.02), borderaxespad=0, columnspacing=0.8,
              handlelength=1.6, handletextpad=0.5)
    despine(ax)


def qc_percent(frame, bin_ms, column):
    values = pd.to_numeric(frame.loc[frame["bin_ms"] == bin_ms, column], errors="coerce").to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid): return "n/a"
    value = "100" if np.all(valid == 1.0) else f"{100 * valid.mean():.2f}"
    return value if len(valid) == len(values) else f"{value} [n={len(valid)}]"


def plot_qc_table(ax, rows):
    ax.axis("off")
    headings = ["Measure", "Order in each cell"] + [f"{b} ms" for b in BIN_MS_LIST]
    display_rows = [[textwrap.fill(str(row[0]), width=26, break_long_words=False, break_on_hyphens=False),
                     textwrap.fill(str(row[1]), width=24, break_long_words=False, break_on_hyphens=False),
                     *row[2:]] for row in rows]
    table = ax.table(cellText=display_rows, colLabels=headings, cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[0.32, 0.28] + [0.40 / len(BIN_MS_LIST)] * len(BIN_MS_LIST))
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_SIZE)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.80")
        cell.set_linewidth(0.5)
        cell.PAD = 0.055
        if row == 0:
            cell.set_facecolor("0.94")
            cell.set_text_props(fontweight="bold")
        if col < 2: cell.get_text().set_ha("left")
    return table


# ------------------------------------------------------------
# Neuron-count sensitivity on the common session cohort
# ------------------------------------------------------------


def plot_neuron_count_effect(ax, effects, metric, ylabel):
    selected = effects[effects["Metric"] == metric]
    offsets = dict(zip(BIN_MS_LIST, np.linspace(-0.65, 0.65, len(BIN_MS_LIST))))
    ax.axhline(0, color="0.45", ls="--", lw=0.9, zorder=0)
    notes = []
    for b in BIN_MS_LIST:
        d = selected[selected["bin_ms"] == b].set_index("n_neurons").loc[NEURON_COUNT_VALUES]
        x = np.asarray(NEURON_COUNT_VALUES, dtype=float) + offsets[b]
        y = d["estimate"].to_numpy(dtype=float)
        ax.plot(x, y, color=BIN_COLORS[b], lw=1.5, zorder=1)
        for xpos, n, (_, row) in zip(x, NEURON_COUNT_VALUES, d.iterrows()):
            estimate, lo, hi = float(row["estimate"]), float(row["ci95_low"]), float(row["ci95_high"])
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(xpos, lo, hi, color=BIN_COLORS[b], lw=1.25)
                ax.hlines([lo, hi], xpos - 0.35, xpos + 0.35, color=BIN_COLORS[b], lw=1.1)
            if np.isfinite(estimate):
                ax.plot(xpos, estimate, marker=NEURON_COUNT_MARKERS[n], linestyle="none", color=BIN_COLORS[b],
                        markerfacecolor="white" if n == NEURON_COUNT_VALUES[0] else BIN_COLORS[b], ms=6, zorder=3)
        if not np.isfinite(y).all(): notes.append(f"{b} ms: unavailable N=" + "/".join(str(n) for n, value in zip(NEURON_COUNT_VALUES, y) if not np.isfinite(value)))
    counts = selected["n_pairs"].to_numpy(dtype=int)
    if len(np.unique(counts)) != 1:
        counts_by_bin = [int(selected.loc[selected["bin_ms"] == b, "n_pairs"].iloc[0]) for b in BIN_MS_LIST]
        notes.insert(0, "n (20 / 50 / 100 ms): " + " / ".join(map(str, counts_by_bin)))
    if notes: ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=NOTE_SIZE)
    ax.set_xticks(NEURON_COUNT_VALUES)
    ax.set_xlim(min(NEURON_COUNT_VALUES) - 3, max(NEURON_COUNT_VALUES) + 3)
    ax.set_xlabel("Number of neurons (N)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls=":", alpha=0.22)
    ax.margins(y=0.15)
    despine(ax)


def plot_neuron_count_correlations(ax, correlations, condition, show_labels=True):
    specs = [(metric, label, b) for metric, label in NEURON_COUNT_CORR_METRICS for b in BIN_MS_LIST]
    y_positions = np.array([len(specs) - 1 - j + 0.5 * (len(NEURON_COUNT_CORR_METRICS) - 1 - j // len(BIN_MS_LIST)) for j in range(len(specs))])
    offsets = dict(zip(NEURON_COUNT_VALUES, np.linspace(0.22, -0.22, len(NEURON_COUNT_VALUES))))
    selected = correlations[correlations["condition"] == condition]
    counts, missing = [], []
    ax.axvline(0, color="0.45", ls="--", lw=0.9, zorder=0)
    for y, (metric, _, b) in zip(y_positions, specs):
        d = selected[(selected["metric"] == metric) & (selected["bin_ms"] == b)].set_index("n_neurons").loc[NEURON_COUNT_VALUES]
        counts.append(int(d["n_sessions"].iloc[0]))
        failed = []
        for n, row in d.iterrows():
            rho, lo, hi = float(row["rho"]), float(row["ci95_low"]), float(row["ci95_high"])
            draw_horizontal_ci(ax, rho, lo, hi, y + offsets[n], BIN_COLORS[b], marker=NEURON_COUNT_MARKERS[n], cap=0.05)
            if not np.isfinite(rho): failed.append(str(n))
        missing.append(failed)
    # Show per-row sample sizes only when they differ; retain missing-value notes.
    for y, n_sessions, failed in zip(y_positions, counts, missing):
        note = f"n={n_sessions}" if len(set(counts)) > 1 else ""
        if failed: note += ("; " if note else "") + "NA: " + "/".join(failed)
        if note: ax.text(1.025, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=NOTE_SIZE)
    for j in range(len(BIN_MS_LIST), len(specs), len(BIN_MS_LIST)):
        ax.axhline((y_positions[j - 1] + y_positions[j]) / 2, color="0.85", lw=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{label}, {b} ms" for _, label, b in specs], fontsize=NOTE_SIZE)
    if not show_labels: ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(y_positions[-1] - 0.65, y_positions[0] + 0.65)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"Partial Spearman $\rho$")
    ax.grid(axis="x", ls=":", alpha=0.2)
    despine(ax)


# ------------------------------------------------------------
# Stimulus examples for Figure 1
# ------------------------------------------------------------


def generate_grating(angle_deg, size=(100, 100), freq_cyc_deg=0.5, phase=0.0):
    angle_rad = np.deg2rad(angle_deg)
    x, y = np.linspace(-5, 5, size[1]), np.linspace(-5, 5, size[0])
    X, Y = np.meshgrid(x, y)
    u = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    return np.sin(2 * np.pi * freq_cyc_deg * u + phase)


def get_natural_movie_frames():
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        manifest = os.path.join(PROJECT_ROOT, "allen_ecephys_cache", "manifest.json")
        cache = EcephysProjectCache.from_warehouse(manifest=manifest)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Reading `\.npy` or `\.npz` file required additional header parsing.*",
                category=UserWarning
            )
            movie_template = cache.get_natural_movie_template(1)
        frame_indices = [200, 213, 226]
        if movie_template is None or len(movie_template) <= max(frame_indices):
            raise ValueError("The natural_movie_one template lacks the requested example frames.")
        frames = []
        for idx in frame_indices:
            frame = np.asarray(movie_template[idx])
            if frame.ndim != 2 or not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
                raise ValueError(f"Movie frame {idx} must be a finite two-dimensional image.")
            h, w = frame.shape
            center_w, half_h = w // 2, h // 2
            if half_h < 1 or center_w - half_h < 0 or center_w + half_h > w:
                raise ValueError(f"Movie frame {idx} cannot support the requested center crop: shape={frame.shape}.")
            frames.append(frame[:, center_w - half_h:center_w + half_h])
        return frames
    except Exception as exc:
        raise RuntimeError("Cannot prepare the real natural_movie_one example frames.") from exc


# ============================================================
# 3. Load data
# ============================================================

PER_SESSION_DF = read_csv("per_session.csv")

SUMMARY_DF = read_csv("summary_stats.csv")

RAW_SUBSAMPLING_DF = read_csv("subsampling_raw_data.csv")

ACF_DF = read_csv("acf_data.csv")

PCA_SUMMARY_DF = read_csv("pca_summary.csv")

TAU_SPECTRUM_DF = read_csv("tau_spectrum_data.csv")

TAU_SUMMARY_CORR_DF = read_csv("tau_summary_dim_correlations.csv")

TAU_SHAPE_CORR_DF = read_csv("tau_spectral_shape_dim_correlations.csv")

TAU_SHAPE_SESSION_DF = read_csv("tau_spectral_shape_per_session.csv")

TAU_TOPK_CORR_DF = read_csv("tau_topk_dim_correlations.csv")

TAU_MODE_CORR_DF = read_csv("tau_mode_by_mode_dim_correlations.csv")

PSTH_PER_DF = read_csv("psth_per_session.csv")

PSTH_SUMMARY_DF = read_csv("psth_summary_stats.csv")

RAW_PSTH_CORR_DF = read_csv("raw_psth_dim_correlations.csv")

RAW_PSTH_COMMON_DF = read_csv("raw_psth_common_per_session.csv")

PSTH_QC_DF = read_csv("psth_qc_summary.csv")

CORR_DF = build_combined_correlation_table()

neuron_effects = read_csv("neuron_count_sensitivity_summary_stats.csv")
neuron_correlations = read_csv("neuron_count_sensitivity_dim_correlations.csv")

movie_frames = get_natural_movie_frames()


# ============================================================
# 4. Plot figures in order
# ============================================================

# ------------------------------------------------------------
# Figure 1: Population geometry
# ------------------------------------------------------------

fig1 = plt.figure(figsize=(10.8, 6.2))
gs1 = fig1.add_gridspec(2, 2, height_ratios=[0.65, 1.25], hspace=0.40, wspace=0.42)
fig1.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.91)

#A
subA = gs1[0, :].subgridspec(1, 2, wspace=0.25)
subAg = subA[0].subgridspec(1, 3, wspace=0.06)
subAm = subA[1].subgridspec(1, 3, wspace=0.06)
axes_g = [fig1.add_subplot(subAg[0, i]) for i in range(3)]
axes_m = [fig1.add_subplot(subAm[0, i]) for i in range(3)]
for ax, phase in zip(axes_g, [0, np.pi / 2, np.pi]):
    ax.imshow(generate_grating(0, phase=phase), cmap="gray", aspect="equal")
    ax.axis("off")
for ax, frame in zip(axes_m, movie_frames):
    ax.imshow(frame, cmap="gray", aspect="equal")
    ax.axis("off")

ax1B = fig1.add_subplot(gs1[1, 0])
ax1C = fig1.add_subplot(gs1[1, 1])

#B
plot_pca_panel(ax1B, bin_ms=100, use_std=True)

#C
plot_condition_metric_across_bins( ax1C, "D_eff", r"Effective dimensionality $D_{eff}$")
ax1C.legend(frameon=False, loc="best")


add_panel_labels(fig1, {"A": axes_g + axes_m, "B": ax1B, "C": ax1C},
                 figure_style={}, panel_styles={})
save_figure(fig1, "Figure1_Geometry")


# ------------------------------------------------------------
# Figure 2: Temporal dynamics across resolutions
# ------------------------------------------------------------

fig2 = plt.figure(figsize=(12.8, 16.4))
fig2.subplots_adjust(left=0.115, right=0.985, bottom=0.055, top=0.95)
gs2 = fig2.add_gridspec(3, 1, height_ratios=[1.0, 4.45, 1.08], hspace=0.32)
gs2A = gs2[0].subgridspec(1, 3, wspace=0.47)
gs2B = gs2[1].subgridspec(4, 3, hspace=0.62, wspace=0.47)
gs2C = gs2[2].subgridspec(1, 3, wspace=0.47)

#A
acf_axes = [
    fig2.add_subplot(gs2A[0, 0]),
    fig2.add_subplot(gs2A[0, 1]),
    fig2.add_subplot(gs2A[0, 2])
]
for i, (ax, b) in enumerate(zip(acf_axes, BIN_MS_LIST)):
    plot_acf_panel(ax, b, show_legend=(i == 0))
    ax.set_title(f"{b} ms")
    if i > 0: ax.set_ylabel("")

#B
tau_specs = [
    ("Exp_Tau_mean_ms", r"Mean $\tau$ (ms)"),
    ("Exp_Tau_median_ms", r"Median $\tau$ (ms)"),
    ("Exp_Tau_max_ms", "Slowest retained\n" + r"$\tau$ (ms)"),
    ("Exp_Tau_min_ms", "Fastest retained\n" + r"$\tau$ (ms)")
]
tau_axes = []
for row_index, (metric, ylabel) in enumerate(tau_specs):
    for column, bin_ms in enumerate(BIN_MS_LIST):
        ax = fig2.add_subplot(gs2B[row_index, column])
        tau_axes.append(ax)
        plot_condition_metric_at_bin(ax, SUMMARY_DF, bin_ms, metric, ylabel if column == 0 else "")
        if row_index == 0: ax.set_title(f"{bin_ms} ms")

#C
spec_axes = [
    fig2.add_subplot(gs2C[0, 0]),
    fig2.add_subplot(gs2C[0, 1]),
    fig2.add_subplot(gs2C[0, 2])
]
for i, (ax, b) in enumerate(zip(spec_axes, BIN_MS_LIST)):
    plot_tau_spectrum_panel(ax, b, show_legend=(i == 0))
    ax.set_title(f"{b} ms")
    if i > 0: ax.set_ylabel("")


add_panel_labels(fig2, {"A": acf_axes, "B": tau_axes, "C": spec_axes},
                 figure_style={}, panel_styles={})
save_figure(fig2, "Figure2_Dynamics")


# ------------------------------------------------------------
# Figure 3: Geometry-dynamics relationship across metrics/scales
# ------------------------------------------------------------

fig3, axes3 = plt.subplots(1, 3, figsize=(12.8, 6.6), sharex=True)
#A
plot_correlation_forest(axes3[0], "Grating")
axes3[0].set_title("Gratings")
#B
plot_correlation_forest(axes3[1], "Movie")
axes3[1].set_title("Movie")
#C
plot_correlation_forest(axes3[2], "Delta")
axes3[2].set_title("Movie - Gratings")
handles3, labels3 = axes3[0].get_legend_handles_labels()
fig3.legend(handles3, labels3, frameon=False, loc="lower center", bbox_to_anchor=(0.61, 0.015), ncol=3)
for ax in axes3[1:]:
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")
fig3.subplots_adjust(left=0.255, right=0.98, bottom=0.21, top=0.94, wspace=0.30)

add_panel_labels(fig3, {"A": axes3[0], "B": axes3[1], "C": axes3[2]},
                 figure_style={}, panel_styles={})
save_figure(fig3, "Figure3_Geometry_Dynamics_Forest")


# ------------------------------------------------------------
# Figure 4: Rank-resolved correlations and log-spectrum slope of unaveraged movie responses
# ------------------------------------------------------------

fig4 = plt.figure(figsize=(12.0, 8.2))
gs4 = fig4.add_gridspec(4, 3, height_ratios=[1.20, 0.08, 1.15, 1.05], hspace=0.40, wspace=0.28)
fig4.subplots_adjust(left=0.115, right=0.98, bottom=0.025, top=0.93)

#A
ax4A = fig4.add_subplot(gs4[0, :])
plot_mode_corr(ax4A, "Movie", show_ci=True, legend=True)

#B
slope_axes = []
for i, bin_ms in enumerate(BIN_MS_LIST):
    ax = fig4.add_subplot(gs4[2, i], sharey=slope_axes[0] if slope_axes else None)
    slope_axes.append(ax)
    stats_ax = fig4.add_subplot(gs4[3, i])
    plot_movie_slope_scatter(ax, bin_ms, stats_ax=stats_ax)
    ax.set_title(f"{bin_ms} ms")
    if i > 0:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)


add_panel_labels(fig4, {"A": ax4A, "B": slope_axes},
                 figure_style={}, panel_styles={})
save_figure(fig4, "Figure4_Spectral_Hierarchy")


# ------------------------------------------------------------
# Extended Data Figure 1-1: PCA curves at 20 and 50 ms
# ------------------------------------------------------------

figS2, axesS2 = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
#A
plot_pca_panel(axesS2[0], bin_ms=20, use_std=True)
axesS2[0].set_title("20 ms")

#B
plot_pca_panel(axesS2[1], bin_ms=50, use_std=True)
axesS2[1].set_title("50 ms")
leg = axesS2[1].get_legend()
if leg is not None: leg.remove()
axesS2[1].set_ylabel("")
figS2.tight_layout(pad=1.3, w_pad=2.0)

add_panel_labels(figS2, {"A": axesS2[0], "B": axesS2[1]},
                 figure_style={}, panel_styles={})
save_figure(figS2, "Figure1-1")


# ------------------------------------------------------------
# Extended Data Figure 1-2: Within-session subsampling variation
# ------------------------------------------------------------

figS1, axesS1 = plt.subplots(2, 3, figsize=(12.0, 6.5), sharey=False)
for col in (1, 2):
    axesS1[0, col].sharey(axesS1[0, 0])
    axesS1[0, col].tick_params(axis="y", labelleft=False)
    if SUBSAMPLING_SHARED_TAU_AXIS:
        axesS1[1, col].sharey(axesS1[1, 0])
        axesS1[1, col].tick_params(axis="y", labelleft=False)
for col, b in enumerate(BIN_MS_LIST):
    per = PER_SESSION_DF[PER_SESSION_DF["bin_ms"] == b]
    raw = RAW_SUBSAMPLING_DF[RAW_SUBSAMPLING_DF["bin_ms"] == b].copy()

    #A
    order_dim = per.sort_values("delta_dim")["session_id"].tolist()
    plot_subsampling_distribution(axesS1[0, col], raw, order_dim, "delta_dim")
    axesS1[0, col].set_title(f"{b} ms")
    axesS1[0, col].axhline(0, color="0.4", ls="--", lw=1)
    axesS1[0, col].set_xticks([])
    axesS1[0, col].set_xlabel("")
    axesS1[0, col].set_ylabel(r"$\Delta D_{eff}$" if col == 0 else "")
    despine(axesS1[0, col])

    #B
    order_tau = per.sort_values("delta_tau")["session_id"].tolist()
    plot_subsampling_distribution(axesS1[1, col], raw, order_tau, "delta_tau")
    axesS1[1, col].axhline(0, color="0.4", ls="--", lw=1)
    axesS1[1, col].set_xticks([])
    axesS1[1, col].set_xlabel("Sessions, sorted by\nmean effect")
    axesS1[1, col].set_ylabel(r"$\Delta$ mean $\tau$ (ms)" if col == 0 else "")
    despine(axesS1[1, col])

figS1.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.92, hspace=0.35, wspace=0.32)

add_panel_labels(figS1, {"A": axesS1[0], "B": axesS1[1]},
                 figure_style={}, panel_styles={})
save_figure(figS1, "Figure1-2")


# ------------------------------------------------------------
# Extended Data Figure 1-3: Neuron-count sensitivity in a common cohort
# ------------------------------------------------------------

figS9 = plt.figure(figsize=(12.4, 8.8))
gsS9 = figS9.add_gridspec(2, 6, height_ratios=[1.0, 1.55], hspace=0.55, wspace=0.85)
figS9.subplots_adjust(left=0.20, right=0.98, bottom=0.11, top=0.86)
axS9A = figS9.add_subplot(gsS9[0, :3])
axS9B = figS9.add_subplot(gsS9[0, 3:])
#A
plot_neuron_count_effect(axS9A, neuron_effects, "D_eff", r"$\Delta D_{eff}$" + "\n(Movie - Gratings)")

#B
plot_neuron_count_effect(axS9B, neuron_effects, "Exp_Tau_mean_ms", r"$\Delta\,\overline{\tau}_{1:20}$ (ms)" + "\n(Movie - Gratings)")
corr_axesS9 = []
#C
ax = figS9.add_subplot(gsS9[1, 0:2], sharey=corr_axesS9[0] if corr_axesS9 else None)
corr_axesS9.append(ax)
plot_neuron_count_correlations(ax, neuron_correlations, "Grating", show_labels=True)
ax.set_title("Gratings")

#D
ax = figS9.add_subplot(gsS9[1, 2:4], sharey=corr_axesS9[0] if corr_axesS9 else None)
corr_axesS9.append(ax)
plot_neuron_count_correlations(ax, neuron_correlations, "Movie", show_labels=False)
ax.set_title("Movie")

#E
ax = figS9.add_subplot(gsS9[1, 4:6], sharey=corr_axesS9[0] if corr_axesS9 else None)
corr_axesS9.append(ax)
plot_neuron_count_correlations(ax, neuron_correlations, "Delta", show_labels=False)
ax.set_title("Movie - Gratings")
bin_handlesS9 = [Line2D([], [], color=BIN_COLORS[b], lw=2, label=f"{b} ms") for b in BIN_MS_LIST]
n_handlesS9 = [Line2D([], [], color="0.30", marker=NEURON_COUNT_MARKERS[n], linestyle="none", ms=6,
                         markerfacecolor="none" if n == NEURON_COUNT_VALUES[0] else "0.30", label=f"N={n}") for n in NEURON_COUNT_VALUES]
figS9.legend(handles=bin_handlesS9, title="Bin width", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.36, 0.99))
figS9.legend(handles=n_handlesS9, title="Neurons per population", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.76, 0.99))

add_panel_labels(figS9, {"A": axS9A, "B": axS9B, "C": corr_axesS9[0], "D": corr_axesS9[1], "E": corr_axesS9[2]},
                 figure_style={"y_pad_points": 5}, panel_styles={})
save_figure(figS9, "Figure1-3")

# ------------------------------------------------------------
# Extended Data Figure 1-4: Response-statistics characterization
# ------------------------------------------------------------

figS6, axesS6 = plt.subplots(1, 2, figsize=(10.4, 4.6))

#A
plot_paired_condition_metric(
    axesS6[0],
    bin_ms=100,
    col_g="g_fano",
    col_m="m_fano",
    ylabel="Fano factor"
)

#B
plot_condition_metric_across_bins(
    axesS6[1],
    "SNR",
    "Population SNR",
    annotate=True
)
axesS6[1].legend(frameon=False, loc="best")

figS6.tight_layout(pad=1.3, w_pad=2.0)

add_panel_labels(figS6, {"A": axesS6[0], "B": axesS6[1]},
                 figure_style={}, panel_styles={})
save_figure(figS6, "Figure1-4")


# ------------------------------------------------------------
# Extended Data Figure 2-1: Unaveraged prediction and compact coverage summary
# ------------------------------------------------------------

figS5 = plt.figure(figsize=(10.0, 6.5))
gsS5 = figS5.add_gridspec(2, 1, height_ratios=[3.1, 1.25], hspace=0.60)
figS5.subplots_adjust(left=0.11, right=0.98, bottom=0.07, top=0.93)
#A
axS5A = figS5.add_subplot(gsS5[0])
plot_ols_prediction_qc(axS5A, PER_SESSION_DF, r"Held-out $R^2$")

#B
stable_cells, session_cells = [], []
for b in BIN_MS_LIST:
    stable_cells.append(" / ".join(qc_percent(PER_SESSION_DF, b, f"{s}_valid_mode_fraction") for s in ("g", "m")))
    counts = []
    for stim in ("g", "m"):
        d = PER_SESSION_DF[PER_SESSION_DF["bin_ms"] == b]
        mask = np.isfinite(pd.to_numeric(d[f"{stim}_valid_mode_fraction"], errors="coerce"))
        counts.append(str(d.loc[mask, "session_id"].nunique()))
    session_cells.append(" / ".join(counts))
rowsS5 = [["Valid stable modes (%)", "Gratings / Movie", *stable_cells],
          ["Sessions (n)", "Gratings / Movie", *session_cells]]
axS5B = figS5.add_subplot(gsS5[1])
plot_qc_table(axS5B, rowsS5)

add_panel_labels(figS5, {"A": axS5A, "B": axS5B},
                 figure_style={}, panel_styles={})
save_figure(figS5, "Figure2-1")


# ------------------------------------------------------------
# Extended Data Figure 4-1: Full mode-by-mode correlation profiles
# ------------------------------------------------------------

figS3, axesS3 = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
#A
plot_mode_corr(axesS3[0], "Grating", show_ci=True, legend=False)
axesS3[0].set_title("Gratings")

#B
plot_mode_corr(axesS3[1], "Delta", show_ci=True, legend=False)
axesS3[1].set_title("Movie - Gratings")
axesS3[1].set_ylabel("")
figS3.tight_layout(pad=1.3, w_pad=2.0, rect=[0, 0, 1, 0.87])
handlesS3, labelsS3 = axesS3[0].get_legend_handles_labels()
figS3.legend(handlesS3, labelsS3, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.0))

add_panel_labels(figS3, {"A": axesS3[0], "B": axesS3[1]},
                 figure_style={}, panel_styles={})
save_figure(figS3, "Figure4-1")


# ------------------------------------------------------------
# Extended Data Figure 4-2: Top-K slowest-mode correlations
# ------------------------------------------------------------

figS4, axesS4 = plt.subplots(1, 3, figsize=(11.8, 4.5), sharey=True)
#A
ax = axesS4[0]
ax.set_title("Gratings")
for b in BIN_MS_LIST:
    df = TAU_TOPK_CORR_DF[(TAU_TOPK_CORR_DF["condition"] == "Grating") & (TAU_TOPK_CORR_DF["bin_ms"] == b)].sort_values("top_k")
    ax.plot(df["top_k"], df["rho"], marker="o", lw=1.8, ms=5, color=BIN_COLORS[b], label=f"{b} ms")
    ax.fill_between(df["top_k"].to_numpy(), df["ci95_low"].to_numpy(), df["ci95_high"].to_numpy(), color=BIN_COLORS[b], alpha=0.10, edgecolor="none")
ax.axhline(0, color="0.4", ls="--", lw=1)
ax.set_xticks(sorted(TAU_TOPK_CORR_DF["top_k"].dropna().unique()))
ax.set_xlabel("Slowest modes included, K")
ax.set_ylabel(r"Partial Spearman $\rho$")
ax.legend(frameon=False)
despine(ax)

#B
ax = axesS4[1]
ax.set_title("Movie")
for b in BIN_MS_LIST:
    df = TAU_TOPK_CORR_DF[(TAU_TOPK_CORR_DF["condition"] == "Movie") & (TAU_TOPK_CORR_DF["bin_ms"] == b)].sort_values("top_k")
    ax.plot(df["top_k"], df["rho"], marker="o", lw=1.8, ms=5, color=BIN_COLORS[b], label=f"{b} ms")
    ax.fill_between(df["top_k"].to_numpy(), df["ci95_low"].to_numpy(), df["ci95_high"].to_numpy(), color=BIN_COLORS[b], alpha=0.10, edgecolor="none")
ax.axhline(0, color="0.4", ls="--", lw=1)
ax.set_xticks(sorted(TAU_TOPK_CORR_DF["top_k"].dropna().unique()))
ax.set_xlabel("Slowest modes included, K")
despine(ax)

#C
ax = axesS4[2]
ax.set_title("Movie - Gratings")
for b in BIN_MS_LIST:
    df = TAU_TOPK_CORR_DF[(TAU_TOPK_CORR_DF["condition"] == "Delta") & (TAU_TOPK_CORR_DF["bin_ms"] == b)].sort_values("top_k")
    ax.plot(df["top_k"], df["rho"], marker="o", lw=1.8, ms=5, color=BIN_COLORS[b], label=f"{b} ms")
    ax.fill_between(df["top_k"].to_numpy(), df["ci95_low"].to_numpy(), df["ci95_high"].to_numpy(), color=BIN_COLORS[b], alpha=0.10, edgecolor="none")
ax.axhline(0, color="0.4", ls="--", lw=1)
ax.set_xticks(sorted(TAU_TOPK_CORR_DF["top_k"].dropna().unique()))
ax.set_xlabel("Slowest modes included, K")
despine(ax)
figS4.tight_layout(pad=1.3, w_pad=2.0)

add_panel_labels(figS4, {"A": axesS4[0], "B": axesS4[1], "C": axesS4[2]},
                 figure_style={}, panel_styles={})
save_figure(figS4, "Figure4-2")


# ------------------------------------------------------------
# Extended Data Figure 4-3: Identity-aligned repeat-average analysis
# ------------------------------------------------------------

s7_effect_rows = PSTH_SUMMARY_DF[PSTH_SUMMARY_DF["Metric"].isin(["D_eff", "Exp_Tau_mean_ms", "Log_spectrum_slope"])]
s7_same_effect_n = s7_effect_rows["n_pairs"].notna().all() and s7_effect_rows["n_pairs"].nunique() == 1
s7_same_corr_n = RAW_PSTH_CORR_DF["n_sessions"].notna().all() and RAW_PSTH_CORR_DF["n_sessions"].nunique() == 1
figS7 = plt.figure(figsize=(12.4, 13.2))
gsS7 = figS7.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 1.0, 0.17, 1.60], hspace=0.62, wspace=0.47)
figS7.subplots_adjust(left=0.17, right=0.94, bottom=0.065, top=0.93)
#A
dim_axesS7 = [figS7.add_subplot(gsS7[0, i]) for i in range(len(BIN_MS_LIST))]
for i, (ax, b) in enumerate(zip(dim_axesS7, BIN_MS_LIST)):
    plot_psth_condition_metric(ax, b, "D_eff", r"PSTH $D_{eff}$" if i == 0 else "", show_n=not s7_same_effect_n)
    ax.set_title(f"{b} ms")

#B
tau_axesS7 = [figS7.add_subplot(gsS7[1, i]) for i in range(len(BIN_MS_LIST))]
for i, (ax, b) in enumerate(zip(tau_axesS7, BIN_MS_LIST)):
    plot_psth_condition_metric(ax, b, "Exp_Tau_mean_ms", r"PSTH mean $\tau$ (ms)" if i == 0 else "", show_n=not s7_same_effect_n)

#C
slope_axesS7 = [figS7.add_subplot(gsS7[2, i]) for i in range(len(BIN_MS_LIST))]
for i, (ax, b) in enumerate(zip(slope_axesS7, BIN_MS_LIST)):
    plot_psth_condition_metric(ax, b, "Log_spectrum_slope", r"Log-spectrum slope $b$" if i == 0 else "", show_n=not s7_same_effect_n)
#D
axS7D = figS7.add_subplot(gsS7[4, 0])
plot_raw_psth_correlation_comparison(axS7D, "Grating", show_labels=True, show_legend=False, show_n=not s7_same_corr_n)
axS7D.set_title("Gratings")

#E
axS7E = figS7.add_subplot(gsS7[4, 1])
plot_raw_psth_correlation_comparison(axS7E, "Movie", show_labels=False, show_legend=False, show_n=not s7_same_corr_n)
axS7E.set_title("Movie")

#F
axS7F = figS7.add_subplot(gsS7[4, 2])
plot_raw_psth_correlation_comparison(axS7F, "Delta", show_labels=False, show_legend=False, show_n=not s7_same_corr_n)
axS7F.set_title("Movie − Grating")
handlesS7 = [Line2D([], [], marker="o", color="0.35", markerfacecolor="none", linestyle="none", label="Unaveraged"),
             Line2D([], [], marker="s", color="0.35", linestyle="none", label="PSTH")]
legendS7 = figS7.add_subplot(gsS7[3, :])
legendS7.axis("off")
legendS7.legend(handles=handlesS7, frameon=False, fontsize=LEGEND_SIZE, ncol=2, loc="center")

add_panel_labels(figS7, {"A": dim_axesS7, "B": tau_axesS7, "C": slope_axesS7,
                  "D": axS7D, "E": axS7E, "F": axS7F},
                 figure_style={}, panel_styles={})
save_figure(figS7, "Figure4-3")


# ------------------------------------------------------------
# Extended Data Figure 4-4: Repeat-average prediction, conditioning and QC table
# ------------------------------------------------------------

figS8 = plt.figure(figsize=(11.8, 8.2))
gsS8 = figS8.add_gridspec(2, 2, height_ratios=[3.0, 2.1], hspace=0.55, wspace=0.37)
figS8.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.05)
#A
axS8A = figS8.add_subplot(gsS8[0, 0])
plot_ols_prediction_qc(axS8A, PSTH_PER_DF, r"Held-out PSTH $R^2$")

#B
axS8B = figS8.add_subplot(gsS8[0, 1])
plot_psth_qc_by_condition(axS8B, "design_condition_number", r"$\log_{10}$ condition number", log10=True)

#C
qc_psth = PSTH_QC_DF[PSTH_QC_DF["representation"] == "PSTH"].set_index("bin_ms")
qc_psth = qc_psth.reindex(BIN_MS_LIST)
common_n = []
for b in BIN_MS_LIST:
    d = RAW_PSTH_COMMON_DF[(RAW_PSTH_COMMON_DF["bin_ms"] == b) & (RAW_PSTH_COMMON_DF["representation"] == "PSTH")]
    common_n.append(d.loc[d["n_common_valid_resamples"] >= 2, "session_id"].nunique())

paired_cells, rank_cells, stable_cells, count_cells = [], [], [], []
for i, b in enumerate(BIN_MS_LIST):
    paired_cells.append(" / ".join(qc_percent(d, b, "paired_spectrum_resample_fraction") for d in (PER_SESSION_DF, PSTH_PER_DF)))
    rank_cells.append(" / ".join(qc_percent(PSTH_PER_DF, b, f"{s}_full_rank") for s in ("g", "m")))
    stable_cells.append(" / ".join(qc_percent(PSTH_PER_DF, b, f"{s}_valid_mode_fraction") for s in ("g", "m")))
    counts = [qc_psth.loc[b, "n_sessions_dim"], qc_psth.loc[b, "n_sessions_mean_tau"], common_n[i]]
    count_cells.append(" / ".join(str(int(v)) if pd.notna(v) and np.isfinite(v) else "n/a" for v in counts))
rowsS8 = [["Complete paired top-20 spectra (%)", "Unaveraged / PSTH", *paired_cells],
          ["PSTH full-rank designs (%)", "Gratings / Movie", *rank_cells],
          ["PSTH valid stable modes (%)", "Gratings / Movie", *stable_cells],
          ["Available sessions (n)", "PSTH geometry / spectrum / common", *count_cells]]
axS8C = figS8.add_subplot(gsS8[1, :])
plot_qc_table(axS8C, rowsS8)

add_panel_labels(figS8, {"A": axS8A, "B": axS8B, "C": axS8C},
                 figure_style={}, panel_styles={})
save_figure(figS8, "Figure4-4")


# ------------------------------------------------------------
# Extended Data Figure 4-5: Alternative log-ratio summaries of timescale structure
# ------------------------------------------------------------

figS10 = plt.figure(figsize=(13.8, 14.0))
gsS10 = figS10.add_gridspec(7, 3, height_ratios=[1.45, 0.25, 1.25, 1.05, 0.15, 1.25, 1.05],
                          hspace=0.40, wspace=0.30)
figS10.subplots_adjust(left=0.245, right=0.98, bottom=0.025, top=0.94)

#A
axS10A = figS10.add_subplot(gsS10[0, 0])
plot_correlation_forest(axS10A, "Grating", metrics=LOG_RATIO_CORR_METRICS)
axS10A.set_title("Gratings")

#B
axS10B = figS10.add_subplot(gsS10[0, 1], sharex=axS10A, sharey=axS10A)
plot_correlation_forest(axS10B, "Movie", metrics=LOG_RATIO_CORR_METRICS)
axS10B.set_title("Movie")
axS10B.tick_params(axis="y", labelleft=False)

#C
axS10C = figS10.add_subplot(gsS10[0, 2], sharex=axS10A, sharey=axS10A)
plot_correlation_forest(axS10C, "Delta", metrics=LOG_RATIO_CORR_METRICS)
axS10C.set_title("Movie − Grating")
axS10C.tick_params(axis="y", labelleft=False)
handlesS10, labelsS10 = axS10A.get_legend_handles_labels()
legendS10 = figS10.add_subplot(gsS10[1, :])
legendS10.axis("off")
legendS10.legend(handlesS10, labelsS10, frameon=False, loc="center", ncol=3)
legendS10.set_position(legendS10.get_position().translated(0, -0.01))

#D
ratio_axesS10, half_ratio_axesS10 = [], []
for i, bin_ms in enumerate(BIN_MS_LIST):
    ax = figS10.add_subplot(gsS10[2, i], sharey=ratio_axesS10[0] if ratio_axesS10 else None)
    ratio_axesS10.append(ax)
    stats_ax = figS10.add_subplot(gsS10[3, i])
    ax.set_position(ax.get_position().translated(0, -0.01))
    stats_ax.set_position(stats_ax.get_position().translated(0, -0.01))
    plot_movie_hierarchy_scatter(ax, bin_ms, stats_ax=stats_ax)
    ax.set_title(f"{bin_ms} ms")
    if i > 0:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

#E
for i, bin_ms in enumerate(BIN_MS_LIST):
    ax = figS10.add_subplot(gsS10[5, i], sharex=ratio_axesS10[i],
                           sharey=half_ratio_axesS10[0] if half_ratio_axesS10 else None)
    half_ratio_axesS10.append(ax)
    stats_ax = figS10.add_subplot(gsS10[6, i])
    ax.set_position(ax.get_position().translated(0, 0.02))
    stats_ax.set_position(stats_ax.get_position().translated(0, 0.02))
    plot_movie_half_ratio_scatter(ax, bin_ms, stats_ax=stats_ax)
    if i > 0:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

add_panel_labels(figS10, {"A": axS10A, "B": axS10B, "C": axS10C,
                         "D": ratio_axesS10, "E": half_ratio_axesS10},
                 figure_style={}, panel_styles={})
save_figure(figS10, "Figure4-5")


print(f"All figures were generated under: {FIGURE_DIR}")
