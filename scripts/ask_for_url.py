import requests


def get_nwb_download_url(session_id):
    api_url = f"http://api.brain-map.org/api/v2/data/EcephysSession/{session_id}.json"

    try:
        response = requests.get(api_url, params={"include": "well_known_files(well_known_file_type)"}, timeout=(10, 30))
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            return f"Query failed: {data.get('msg', 'Unsuccessful API response')}"
        records = data.get("msg", [])
        if not records:
            return "No matching session found."
        files = [f for f in records[0].get("well_known_files", [])
                 if f.get("well_known_file_type", {}).get("name") == "EcephysNwb"]
        if len(files) != 1:
            return f"Expected one EcephysNwb session file; found {len(files)}."
        return f"http://api.brain-map.org/api/v2/well_known_file_download/{files[0]['id']}"

    except Exception as e:
        return f"Query failed: {e}"

# sessions want to download
TARGET_SESSIONS = [715093703, 719161530, 721123822, 732592105, 737581020, 743475441, 750332458, 750749662, 751348571,
               754312389, 754829445, 755434585, 756029989, 757216464, 757970808, 758798717, 759883607, 760345702,
               760693773, 762120172, 762602078, 763673393, 791319847, 797828357, 798911424, 799864342]

for sess in TARGET_SESSIONS:
    url = get_nwb_download_url(sess)
    print(f"{sess}: {url}")