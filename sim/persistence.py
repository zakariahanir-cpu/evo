import json
import os
from .simulation import Simulation

STATE_PATH = os.path.join("state", "simulation_state.json")


def save_state(sim: Simulation, path=STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sim.to_dict(), f)


def load_state(path=STATE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Simulation.from_dict(d)
