# app/utils.py
import re
import json
from typing import List, Dict, Optional, Any

# -----------------------
# Text / JSON utilities
# -----------------------


def normalize_radius(radius, unit):
    if unit.lower() in ["earth", "re", "earth radii"]:
        return radius / 11.2
    return radius  # already in Jupiter radii

def planet_to_text(obj: Dict[str, Any]) -> str:
    """
    Turn a planet dict into natural-language sentences suitable for embeddings.
    Filters out None or missing values gracefully.
    """
    name = obj.get("planet_name") or obj.get("name") or "Unnamed Planet"

    profile = obj.get("planet_profile") or {}
    star = obj.get("host_star") or {}
    disc = obj.get("discovery") or {}
    env = obj.get("environment") or {}

    parts = []

    radius = profile.get('radius_earth_radii')
    mass = profile.get('mass_earth_masses')
    if radius is not None and mass is not None:
        parts.append(f"The planet {name} has a radius of {normalize_radius(radius, 'earth')} Jupiter radii and a mass of {mass} Earth masses.")

    orbital_period = profile.get("orbital_period_days")
    semi_major_axis = profile.get("semi_major_axis_au")
    if orbital_period is not None and semi_major_axis is not None:
        parts.append(f"It orbits its star every {orbital_period} days at a distance of {semi_major_axis} AU.")

    star_name = star.get("name")
    spectral_type = star.get("spectral_type")
    temperature_k = star.get("temperature_k")
    if star_name or spectral_type or temperature_k:
        star_name = star_name or "Unnamed Star"
        spectral_type = spectral_type or "unknown type"
        temperature_k_str = f"{temperature_k} K" if temperature_k is not None else "unknown temperature"
        parts.append(f"The host star {star_name} is a {spectral_type} star with a temperature of {temperature_k_str}.")

    disc_year = disc.get("year")
    disc_method = disc.get("method")
    disc_facility = disc.get("facility")
    if disc_year or disc_method or disc_facility:
        disc_year = disc_year or "an unknown year"
        disc_method = disc_method or "an unknown method"
        disc_facility = disc_facility or "an unknown facility"
        parts.append(f"This planet was discovered in {disc_year} using the {disc_method} method at {disc_facility}.")

    equilibrium_temp = env.get("equilibrium_temperature_k")
    distance_pc = env.get("distance_pc")
    if equilibrium_temp is not None or distance_pc is not None:
        eq_temp_str = f"{equilibrium_temp} K" if equilibrium_temp is not None else "unknown temperature"
        dist_str = f"{distance_pc} parsecs away" if distance_pc is not None else "unknown distance"
        parts.append(f"The equilibrium temperature is {eq_temp_str} and it is {dist_str}.")

    return " ".join(parts)


def chunk_text(text: str, max_len: int = 350) -> List[str]:
    """
    Sentence-aware chunker.
    Splits by sentence (naive using period) and aggregates until max_len reached.
    Keeps chunks semantically coherent (not raw word slices).
    """
    if not text:
        return []

    # Basic sentence split — good enough for most plain text files
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks: List[str] = []
    cur = ""

    for s in sentences:
        # if adding this sentence stays within size, append; otherwise start new chunk
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s

    if cur:
        chunks.append(cur.strip())

    return chunks

# -----------------------
# Loading & previewing
# -----------------------
def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_json_folder(folder: str) -> List[Dict[str, Any]]:
    """
    Load all .json files (and lists inside them) from a folder and return a flat list of dicts.
    Use for structured datasets like exoplanets, blackholes, etc.
    """
    import os
    all_objs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith(".json"):
                p = os.path.join(root, fn)
                data = load_json_file(p)
                if isinstance(data, list):
                    all_objs.extend(data)
                elif isinstance(data, dict):
                    # If dict contains a top-level list key, try to discover it
                    # else append the dict itself
                    appended = False
                    for k, v in data.items():
                        if isinstance(v, list):
                            all_objs.extend(v)
                            appended = True
                            break
                    if not appended:
                        all_objs.append(data)
    return all_objs

def preview_chunks(data_list: List[Dict[str, Any]], max_samples: int = 3, max_len: int = 400):
    """
    Print a small preview of chunks for the first `max_samples` objects.
    Good for QA before indexing.
    """
    for obj in data_list[:max_samples]:
        text = planet_to_text(obj) if isinstance(obj, dict) else str(obj)
        chunks = chunk_text(text, max_len=max_len)

        print(f"\n======== {obj.get('planet_name', obj.get('name', 'Item'))} ========\n")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i} ({len(chunk)} chars) ---\n{chunk}\n")

# -----------------------
# Helpers: year extraction & normalization
# -----------------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def extract_year(query: str) -> Optional[int]:
    """Return the first 4-digit year (1900-2099) found in the query, or None."""
    if not query:
        return None
    m = _YEAR_RE.search(query)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None

def normalize_discovery_years(objects: List[Dict[str, Any]]) -> None:
    """
    In-place normalization: ensure discovery.year is an int when possible.
    Converts strings like '2015 ' -> 2015; leaves missing years as None.
    """
    for o in objects:
        disc = o.get("discovery")
        if not disc:
            continue
        year = disc.get("year")
        if isinstance(year, int):
            continue
        if isinstance(year, str):
            year = year.strip()
            if year.isdigit():
                disc["year"] = int(year)
            else:
                # try to extract a year from the string
                y = _YEAR_RE.search(year)
                disc["year"] = int(y.group(0)) if y else None
        else:
            # anything else -> set to None explicitly for clarity
            disc["year"] = None
