import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import gc
import warnings


warnings.filterwarnings("ignore", module=r"^allensdk(?:\.|$)")
warnings.filterwarnings("ignore", module=r"^hdmf(?:\.|$)")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'allen_ecephys_cache', 'manifest.json')
CACHE = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)

print("=" * 60)
print("filter sessions that meet criteria(s)")
print("=" * 60)

SESSIONS_TABLE = CACHE.get_session_table()
CANDIDATE_SESSION_IDS = SESSIONS_TABLE.index.values
print(f"{len(CANDIDATE_SESSION_IDS)} session(s) pending further inspection. Starting detailed check.")

MIN_FIRING_RATE = 0.5      # minimum firing rate threshold (Hz)
HARD_FLOOR = [40, 50, 60]  # neuron count threshold

valid_sessions = {}
session_valid_units = {}
global_N = {}

for floor in HARD_FLOOR:
    valid_sessions[floor] = []
    session_valid_units[floor] = {}
    global_N[floor] = {}

for sess_id in CANDIDATE_SESSION_IDS:
    try:
        session = CACHE.get_session_data(sess_id)

        # 1. check if required stimuli exist
        stim_names = session.stimulus_presentations['stimulus_name'].unique()
        if 'drifting_gratings' not in stim_names or 'natural_movie_one' not in stim_names:
            print(f"Session {sess_id}: The type of stimulus does not match.")
            continue

        # 2. check number of active neurons in VISp
        units = session.units
        visp_units = units[units['ecephys_structure_acronym'] == 'VISp']
        valid_units = visp_units[visp_units['firing_rate'] >= MIN_FIRING_RATE]
        n_valid = len(valid_units)

        for floor in HARD_FLOOR:
            if n_valid >= floor:
                valid_sessions[floor].append(sess_id)
                session_valid_units[floor][sess_id] = valid_units.index.values
        print(f"Session {sess_id}: {n_valid} active VISp cells.")
        del session
        gc.collect()

    except Exception as e:
        print(f"Session {sess_id}: excluded. data loading error ({e}).")


for floor in HARD_FLOOR:
    if len(valid_sessions[floor]) != 0:
        global_N[floor] = min([len(session_valid_units[floor][sid]) for sid in valid_sessions[floor]])
    else:
        global_N[floor] = None

print("\n" + "=" * 60)
for floor in HARD_FLOOR:
    print(f"{len(valid_sessions[floor])} session(s) met the criteria {floor}: {valid_sessions[floor]}.")
    if global_N[floor] is not None:
        print(f"The minimum number of qualified neurons in the group is {global_N[floor]}")
    print("\n" + "-" * 60)