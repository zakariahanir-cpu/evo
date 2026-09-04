import numpy as np
from .world import World
from .agent import Agent, REPRO_COOLDOWN, REPRO_ENERGY_COST, REPRO_MIN_ENERGY
from . import world as W

TICKS_PER_GENERATION = 400
MIN_POPULATION = 2
MAX_POPULATION = 40


class Simulation:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.world = World(seed=seed)
        self.generation = 0
        self.next_id = 0
        self.agents = []
        self._spawn_initial_pair()

    def _new_id(self):
        self.next_id += 1
        return self.next_id

    def _spawn_initial_pair(self):
        for sex in ("M", "F"):
            x = int(self.rng.integers(0, self.world.width))
            y = int(self.rng.integers(0, self.world.height))
            self.agents.append(Agent(self._new_id(), sex, x, y, rng=self.rng))

    def _positions_set(self):
        return {(a.x, a.y) for a in self.agents if a.alive}

    def run_generation(self, ticks=TICKS_PER_GENERATION):
        births, deaths, food_eaten, wood_collected, shelters_built = 0, 0, 0, 0, 0
        pending_reproductions = []

        for _ in range(ticks):
            self.world.step(self.rng)
            positions = self._positions_set()

            for agent in self.agents:
                if not agent.alive:
                    continue
                obs = agent.build_observation(self.world, positions)
                action = agent.decide(obs)
                result = agent.apply_action(action, self.world)
                if result == "eat":
                    food_eaten += 1
                elif result == "collect":
                    wood_collected += 1
                elif result == "build":
                    shelters_built += 1
                elif result == "reproduce_request":
                    pending_reproductions.append(agent)
                is_shelter_here = self.world.get(agent.x, agent.y) == W.SHELTER
                agent.step_body(self.world, self.rng, is_shelter_here)

            # معالجة طلبات التكاثر: يشترط وجود شريك من الجنس الآخر بجوار الفرد
            handled = set()
            for a in pending_reproductions:
                if not a.alive or a.id in handled or a.repro_cooldown > 0:
                    continue
                if a.energy < REPRO_MIN_ENERGY:
                    continue
                partner = self._find_adjacent_partner(a, handled)
                if partner is None:
                    continue
                if len(self.agents) >= MAX_POPULATION:
                    continue
                weights = self._breed(a, partner)
                sex = "M" if self.rng.random() < 0.5 else "F"
                child = Agent(self._new_id(), sex, a.x, a.y,
                               brain=None, rng=self.rng, born_gen=self.generation)
                from .brain import Brain
                child.brain = Brain(weights=weights)
                self.agents.append(child)
                a.energy -= REPRO_ENERGY_COST
                partner.energy -= REPRO_ENERGY_COST
                a.repro_cooldown = REPRO_COOLDOWN
                partner.repro_cooldown = REPRO_COOLDOWN
                a.children += 1
                partner.children += 1
                handled.add(a.id)
                handled.add(partner.id)
                births += 1

            # إزالة الموتى الآن (بعد تسجيلهم)
            newly_dead = [a for a in self.agents if not a.alive]
            deaths += len(newly_dead)
            self.agents = [a for a in self.agents if a.alive]

            if not self.agents:
                break

        # لو انقرضوا أو قل العدد كثيرًا: نضخ أفرادًا جددًا عشوائيين لإنقاذ الاستمرارية
        rescued = 0
        while len(self.agents) < MIN_POPULATION:
            males = sum(1 for a in self.agents if a.sex == "M")
            females = sum(1 for a in self.agents if a.sex == "F")
            sex = "M" if males <= females else "F"
            x = int(self.rng.integers(0, self.world.width))
            y = int(self.rng.integers(0, self.world.height))
            self.agents.append(Agent(self._new_id(), sex, x, y, rng=self.rng,
                                      born_gen=self.generation))
            rescued += 1

        stats = {
            "generation": self.generation,
            "population": len(self.agents),
            "births": births,
            "deaths": deaths,
            "rescued_random_spawns": rescued,
            "food_eaten": food_eaten,
            "wood_collected": wood_collected,
            "shelters_built": shelters_built,
            "avg_energy": float(np.mean([a.energy for a in self.agents])) if self.agents else 0.0,
            "avg_age": float(np.mean([a.age for a in self.agents])) if self.agents else 0.0,
            "oldest_age": int(max([a.age for a in self.agents])) if self.agents else 0,
            "males": sum(1 for a in self.agents if a.sex == "M"),
            "females": sum(1 for a in self.agents if a.sex == "F"),
        }
        self.generation += 1
        return stats

    def _find_adjacent_partner(self, agent, handled):
        for other in self.agents:
            if (other.id == agent.id or not other.alive or other.sex == agent.sex
                    or other.id in handled or other.repro_cooldown > 0
                    or other.energy < REPRO_MIN_ENERGY):
                continue
            dx = abs(other.x - agent.x)
            dy = abs(other.y - agent.y)
            if dx <= 1 and dy <= 1:
                return other
        return None

    def _breed(self, a, b):
        from .brain import Brain
        return Brain.crossover(a.brain, b.brain, self.rng)

    # ---------- الحفظ والاستكمال ----------
    def to_dict(self):
        return {
            "generation": self.generation,
            "next_id": self.next_id,
            "world": self.world.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, d):
        sim = cls.__new__(cls)
        sim.generation = d["generation"]
        sim.next_id = d["next_id"]
        sim.world = World.from_dict(d["world"])
        sim.agents = [Agent.from_dict(ad) for ad in d["agents"]]
        sim.rng = np.random.default_rng()
        sim.rng.bit_generator.state = d["rng_state"]
        return sim
