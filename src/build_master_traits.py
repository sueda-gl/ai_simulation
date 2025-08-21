# src/build_master_traits.py
import yaml, pprint, textwrap
from pathlib import Path

REQ_PATH = Path(__file__).resolve().parents[1] / "config" / "trait_requirements.yaml"

def get_master_trait_list():
    req = yaml.safe_load(REQ_PATH.read_text())
    traits = set()
    for decision, cols in req.items():
        for col in cols:
            if not isinstance(col, str):
                raise ValueError(f"{decision} has non-string entry: {col}")
            if col.strip().lower() == "placeholder_trait":
                continue                 # ignore placeholders
            traits.add(col)
    return sorted(traits)

if __name__ == "__main__":
    master = get_master_trait_list()
    print("Master-trait list ("+str(len(master))+"):")
    print(textwrap.fill(", ".join(master), width=90))