# utils.py
import json
from typing import Any, Dict, List
import re 

def load_json_folder(folder: str) -> List[Dict[str, Any]]:
    import os
    final = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith(".json"):
                path = os.path.join(root, fn)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            final.append(data)
                        elif isinstance(data, list):
                            final.extend(data)
                    except Exception:
                        pass
    return final

def planet_text(obj: Dict[str, Any]) -> str:
    name = obj.get("planet_name", "Unknown Planet")
    p = obj.get("planet_profile", {})
    s = obj.get("host_star", {})
    d = obj.get("discovery", {})
    e = obj.get("environment", {})

    return (
        f"Planet {name}. "
        f"Radius: {p.get('radius_earth_radii')}. "
        f"Mass: {p.get('mass_earth_masses')}. "
        f"Orbital period: {p.get('orbital_period_days')}. "
        f"Semi-major axis: {p.get('semi_major_axis_au')}. "
        f"Star: {s.get('name')}. "
        f"Discovery year: {d.get('year')}. "
        f"Equilibrium temp: {e.get('equilibrium_temperature_k')}. "
        f"Distance: {e.get('distance_pc')} pc."
    )



def chunk_text(text: str, max_len: int = 350) -> List[str]:
    """
    Splits text into chunks of roughly max_len characters, keeping sentences intact.
    """
    if not text:
        return []

    # naive sentence split
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    current_chunk = ""

    for s in sentences:
        if len(current_chunk) + len(s) + 1 <= max_len:
            current_chunk = (current_chunk + " " + s).strip() if current_chunk else s
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = s

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
