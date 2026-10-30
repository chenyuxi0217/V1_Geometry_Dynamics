import os
import numpy as np
import pandas as pd
import warnings

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

warnings.filterwarnings("ignore", module=r"^allensdk(?:\.|$)")
warnings.filterwarnings("ignore", module=r"^hdmf(?:\.|$)")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "allen_ecephys_cache", "manifest.json")

CACHE = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)

SESSION_IDS = [
    715093703, 719161530, 721123822, 732592105, 737581020, 743475441,
    750332458, 750749662, 751348571, 754312389, 754829445, 755434585,
    756029989, 757216464, 757970808, 758798717, 759883607, 760345702,
    760693773, 762120172, 762602078, 763673393, 791319847, 797828357,
    798911424, 799864342
]

EXPECTED_ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

records = []

for i, session_id in enumerate(SESSION_IDS):
    print(f"[{i + 1}/{len(SESSION_IDS)}] Session {session_id}")

    session = CACHE.get_session_data(session_id)
    stim = session.stimulus_presentations
    g = stim[stim["stimulus_name"] == "drifting_gratings"].copy()

    g["orientation"] = pd.to_numeric(g["orientation"], errors="coerce")
    g["temporal_frequency"] = pd.to_numeric(g["temporal_frequency"], errors="coerce")
    g = g.dropna(subset=["orientation", "temporal_frequency"])

    counts = g.groupby(["orientation", "temporal_frequency"]).size().reset_index(name="n_trials")

    for _, row in counts.iterrows():
        records.append({
            "session_id": session_id,
            "orientation": float(row["orientation"]),
            "temporal_frequency": float(row["temporal_frequency"]),
            "n_trials": int(row["n_trials"])
        })

counts_df = pd.DataFrame(records)

# ------------------------------------------------------------
# Cross-session summary for each temporal frequency
# ------------------------------------------------------------

summary_rows = []

for tf in sorted(counts_df["temporal_frequency"].unique()):
    tf_df = counts_df[counts_df["temporal_frequency"] == tf]

    session_min_counts = []
    complete_sessions = 0

    for session_id in SESSION_IDS:
        s = tf_df[tf_df["session_id"] == session_id]
        count_map = dict(zip(s["orientation"], s["n_trials"]))

        counts_8 = [count_map.get(float(ori), 0) for ori in EXPECTED_ORIENTATIONS]

        if all(n > 0 for n in counts_8):
            complete_sessions += 1

        session_min_counts.append(min(counts_8))

    summary_rows.append({
        "temporal_frequency": tf,
        "complete_sessions": complete_sessions,
        "total_sessions": len(SESSION_IDS),
        "min_repeats_across_all_sessions_and_orientations": min(session_min_counts),
        "median_session_min_repeats": np.median(session_min_counts),
        "mean_session_min_repeats": np.mean(session_min_counts),
        "max_session_min_repeats": max(session_min_counts)
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("TEMPORAL-FREQUENCY AVAILABILITY SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))