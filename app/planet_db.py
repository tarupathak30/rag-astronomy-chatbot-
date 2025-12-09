from pandas import json_normalize
import pandas as pd

ATTRIBUTE_MAP = {
    "radius": "radius_earth_radii",
    "mass": "mass_earth_masses",
    "year": "discovery_year",
    "density": "density",
}

class PlanetDB:
    def __init__(self, data):
        self.df = json_normalize(data, sep="_")
        self.df.columns = [col.replace("planet_profile_", "") for col in self.df.columns]
        self.df.columns = [col.replace("environment_", "") for col in self.df.columns]
        self.df.columns = [col.replace("host_star_", "star_") for col in self.df.columns]

        numeric = [
            "radius_earth_radii", "mass_earth_masses",
            "orbital_period_days", "semi_major_axis_au",
            "eccentricity", "equilibrium_temperature_k",
            "insolation_earth_flux", "distance_pc"
        ]
        for col in numeric:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    @staticmethod
    def normalize_attr(attr):
        mapping = {
            "radius": "radius_earth_radii",
            "planet_radius": "radius_earth_radii",
            "radii": "radius_earth_radii",
            "size": "radius_earth_radii",
            "mass": "mass_earth_masses",
            "planet_mass": "mass_earth_masses",
        }
        return mapping.get(attr, attr)

    def rank(self, intent):
        df = self.df.copy()
        attr = self.normalize_attr(intent.get("attribute"))

        attr = ATTRIBUTE_MAP.get(attr, attr)

        if attr not in df.columns:
            raise KeyError(f"Attribute '{attr}' not found in DataFrame. Available: {df.columns}")

        if "year_filter" in intent:
            df = df[df["discovery_year"] >= intent["year_filter"]]

        if "mass_gt" in intent:
            df = df[df[ATTRIBUTE_MAP["mass"]] > intent["mass_gt"]]

        if "mass_lt" in intent:
            df = df[df[ATTRIBUTE_MAP["mass"]] < intent["mass_lt"]]

        if intent.get("agg") == "max":
            df = df.sort_values(attr, ascending=False)
        elif intent.get("agg") == "min":
            df = df.sort_values(attr, ascending=True)

        limit = intent.get("limit", 10)
        return df.head(limit).to_dict(orient="records")
