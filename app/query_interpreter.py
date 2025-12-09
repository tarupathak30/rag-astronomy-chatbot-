import re
from typing import List, Dict, Any, Optional

NUM_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20
    }

class QueryInterpreter:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data  # your list of planet dicts
    
    

    def parse_intent(self, q: str) -> Dict[str, Any]:
        q = q.lower()
        intent = {}

        # 1. attribute detection
        ATTRS = ["radius", "mass", "orbital_period", "distance", "year"]
        intent["attribute"] = next((attr for attr in ATTRS if attr in q), None)

        # default attribute only if none found
        if not intent["attribute"]:
            intent["attribute"] = "radius"

        # 2. aggregation detection
        if re.search(r"\b(top|largest|max|biggest|maximum)\b", q):
            intent["agg"] = "max"
        elif re.search(r"\b(smallest|min|minimum|tiny)\b", q):
            intent["agg"] = "min"
        elif re.search(r"\b(average|mean)\b", q):
            intent["agg"] = "avg"
        elif "count" in q:
            intent["agg"] = "count"
        else:
            intent["agg"] = "list"

        # 3. plural + limit detection
        num = re.search(r"(top|largest|biggest|min|max)\s*(\d+)", q)
        if num:
            intent["plural"] = True
            intent["limit"] = int(num.group(2))
        else:
            intent["plural"] = bool(re.search(r"\b(top|largest|biggest|list|show)\b", q))

        # fallback limit if plural with no number:
        if intent["plural"] and "limit" not in intent:
            intent["limit"] = 10

        # 4. filters
        if year := re.search(r"(?:after|since|from)\s*(\d{4})", q):
            intent["year_filter"] = int(year.group(1))

        if mass_gt := re.search(r"mass\s*>\s*([\d.]+)", q):
            intent["mass_gt"] = float(mass_gt.group(1))

        if mass_lt := re.search(r"mass\s*<\s*([\d.]+)", q):
            intent["mass_lt"] = float(mass_lt.group(1))

        return intent

    
    def apply_filters(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        filtered = self.data
        
        if "year_filter" in intent:
            filtered = [p for p in filtered if p.get("discovery", {}).get("year", 0) >= intent["year_filter"]]
        
        if "mass_filter" in intent:
            filtered = [p for p in filtered if p.get("planet_profile", {}).get("mass_earth_masses", 0) > intent["mass_filter"]]
        
        return filtered
    
    def aggregate(self, filtered: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        attr_map = {
            "radius": ["radius_earth_radii", "radius_jupiter_radii"],  # allow both
            "mass": ["mass_earth_masses", "mass_jupiter_masses"],
            "period": ["orbital_period_days"],
            "discovery": ["year"],
            "year": ["year"]
        }

        
        attr = attr_map.get(intent.get("attribute"))
        agg = intent.get("agg")
        
        if not attr or not filtered:
            return "No relevant data found."
        
        vals = []
        for p in filtered:
            # Normalize units: convert Jupiter radii to Earth radii if present
            if intent.get("attribute") == "radius":
                r_e = p.get("planet_profile", {}).get("radius_earth_radii")
                r_j = p.get("planet_profile", {}).get("radius_jupiter_radii")

            if r_e:
                val = r_e
            elif r_j:
                val = r_j * 11.21  # convert
            else:
                val = None

            if val is not None:
                vals.append((val, p))
        
        if not vals:
            return "No data with specified attribute."
        
        if agg == "max":
            max_val, planet = max(vals, key=lambda x: x[0])
            return f"The planet with the largest {intent['attribute']} is {planet.get('planet_name', 'Unknown')} ({max_val})."
        elif agg == "min":
            min_val, planet = min(vals, key=lambda x: x[0])
            return f"The planet with the smallest {intent['attribute']} is {planet.get('planet_name', 'Unknown')} ({min_val})."
        elif agg == "avg":
            avg_val = sum(x[0] for x in vals) / len(vals)
            return f"The average {intent['attribute']} is {avg_val:.2f}."
        elif agg == "count":
            return f"There are {len(filtered)} planets matching your criteria."
        else:
            # list top 5
            top5 = sorted(vals, key=lambda x: x[0], reverse=True)[:5]
            names = ", ".join(p.get("planet_name", "Unknown") for _, p in top5)
            return f"Top 5 planets by {intent['attribute']}: {names}."
    
    def answer(self, q: str) -> str:
        intent = self.parse_intent(q)
        filtered = self.apply_filters(intent)
        return self.aggregate(filtered, intent)
