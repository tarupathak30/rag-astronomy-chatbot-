# comparator.py

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

ATTRIBUTE_MAP = {
    "radius": ["radius_earth_radii"],
    "mass": ["mass_earth_masses"],
    "orbital_period": ["orbital_period_days"],
    "period": ["orbital_period_days"],
    "temperature": ["equilibrium_temperature_k"],
    "equilibrium_temperature": ["equilibrium_temperature_k"],
    "distance": ["distance_pc"],
}

JUPITER_TO_EARTH_RADIUS = 11.21
JUPITER_TO_EARTH_MASS = 317.8


class Comparator:
    def __init__(self, data: List[Dict[str, Any]]):
        self.planets = data
        self.raw = data or []
        self.df = self._build_df(self.raw)

    # safe converter
    def _safe(self, v):
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _prefer_column(self, row, col1, col2, multiplier):
        v1 = row.get(col1)
        v2 = row.get(col2)

        if v1 is not None and not pd.isna(v1):
            return v1
        if v2 is not None and not pd.isna(v2):
            return v2 * multiplier
        return None

    def _build_df(self, raw: list) -> pd.DataFrame:
        rows = []
        for obj in raw:
            p = obj.get("planet_profile", {})
            e = obj.get("environment", {})
            s = obj.get("host_star", {})
            d = obj.get("discovery", {})

            row = {
                "planet_name": obj.get("planet_name") or obj.get("name"),
                "radius_earth_radii": self._to_float(p.get("radius_earth_radii")),
                "radius_jupiter_radii": self._to_float(p.get("radius_jupiter_radii")),
                "mass_earth_masses": self._to_float(p.get("mass_earth_masses")),
                "mass_jupiter_masses": self._to_float(p.get("mass_jupiter_masses")),
                "orbital_period_days": self._to_float(p.get("orbital_period_days")),
                "semi_major_axis_au": self._to_float(p.get("semi_major_axis_au")),
                "eccentricity": self._to_float(p.get("eccentricity")),
                "equilibrium_temperature_k": self._to_float(e.get("equilibrium_temperature_k")),
                "distance_pc": self._to_float(e.get("distance_pc") or obj.get("distance_pc")),
                "discovery_year": self._to_int(d.get("year")),
                "host_star_name": s.get("name"),
                "host_star_luminosity": self._to_float(s.get("luminosity_lsun")),
            }

            # flux proxy from luminosity + distance
            L = row["host_star_luminosity"]
            a = row["semi_major_axis_au"]

            flux = np.nan
            if L is not None and a is not None and a > 0:
                flux = L / (a**2)

            row["stellar_flux_proxy"] = flux

            rows.append(row)

        df = pd.DataFrame(rows)

        # Normalize radius/mass
        df["radius_earth_radii"] = df.apply(
            lambda r: self._prefer_column(
                r, "radius_earth_radii", "radius_jupiter_radii", JUPITER_TO_EARTH_RADIUS
            ),
            axis=1,
        )

        df["mass_earth_masses"] = df.apply(
            lambda r: self._prefer_column(
                r, "mass_earth_masses", "mass_jupiter_masses", JUPITER_TO_EARTH_MASS
            ),
            axis=1,
        )

        return df

    @staticmethod
    def _to_float(v):
        try:
            return float(v)
        except:
            return np.nan

    @staticmethod
    def _to_int(v):
        try:
            return int(v)
        except:
            return np.nan

    # ---------------- LOOKUP ENGINE --------------------

    def lookup(self, query: str):
        """
        Main structured query dispatcher.
        Supports:
        - fastest / shortest orbit
        - largest radius
        - most massive
        - hottest
        - highest stellar flux
        - comparison between a given list of planets
        """
        q = query.lower()

        # Detect explicit planet list like: [WASP-12 b, KELT-9 b]
        planet_list = self._extract_planet_list(q)

        # Pick the right attribute based on keywords
        if "fastest" in q or "shortest" in q:
            attr = "orbital_period_days"
            asc = True
        elif "largest" in q and "radius" in q:
            attr = "radius_earth_radii"
            asc = False
        elif "massive" in q or "heaviest" in q:
            attr = "mass_earth_masses"
            asc = False
        elif "hottest" in q or "highest temperature" in q:
            attr = "equilibrium_temperature_k"
            asc = False
        elif "flux" in q or "stellar flux" in q or "receives" in q:
            attr = "stellar_flux_proxy"
            asc = False
        else:
            return None

        # If user gave a planet list → filter only those
        return self._rank_by(attribute=attr, ascending=asc, planets=planet_list, limit=1)


    # ---------------- NUMERIC HELPERS -----------------

    def _numeric_top(self, col, ascending=True):
        df = self.df.dropna(subset=[col])
        if df.empty:
            return None

        row = df.sort_values(col, ascending=ascending).iloc[0]
        return {
            "match_type": "numeric",
            "attribute": col,
            "value": float(row[col]),
            "planet_name": row["planet_name"],
        }

    def _highest_flux(self):
        best = None
        best_flux = -1

        for p in self.raw:
            s = p.get("host_star", {})
            prof = p.get("planet_profile", {})

            R_star = s.get("radius_solar_radii")
            T_star = s.get("temperature_k") or s.get("teff")
            a = prof.get("semi_major_axis_au")

            try:
                R_star = float(R_star)
                T_star = float(T_star)
                a = float(a)
            except:
                continue

            if a <= 0:
                continue

            # Stefan-Boltzmann proportionality
            flux = (R_star**2) * (T_star**4) / (a**2)

            if flux > best_flux:
                best_flux = flux
                best = p

        if best is None:
            return None

        return {
            "match_type": "numeric",
            "attribute": "stellar_flux_proxy",
            "value": best_flux,
            "planet": best,
        }

    def _rank_by(self, column: str, ascending: bool = False, planets: list = None, limit: int = None):
        df = self.df.copy()

        # filter
        if planets:
            df = df[df["planet_name"].isin(planets)]

        df = df[df[column].notna()]
        if df.empty:
            return []

        df_sorted = df.sort_values(column, ascending=ascending)
        if limit:
            df_sorted = df_sorted.head(limit)

        return df_sorted.to_dict(orient="records")



    def _extract_planet_list(self, q: str):
        """
        Extracts planet names inside [ ... ] from user query.
        Example:
            'between [WASP-12 b, KELT-9 b]' → ['WASP-12 b','KELT-9 b']
        """
        import re

        match = re.search(r"\[(.*?)\]", q)
        if not match:
            return None

        inside = match.group(1)
        names = [p.strip() for p in inside.split(",") if p.strip()]

        # Only keep planets that actually exist in your DB
        valid = set(self.df['planet_name'])
        return [p for p in names if p in valid]
