"""
الفرد: له جسد (طاقة/صحة/عمر)، إدراك محدود (رؤية + ذاكرة قصيرة)،
ودماغ (Brain) هو نفسه جيناته القابلة للوراثة والطفرة.
"""
import numpy as np
from .brain import Brain, VISION_RADIUS, MOVE_DELTAS
from . import world as W

MAX_ENERGY = 100.0
MAX_HEALTH = 100.0
MAX_AGE = 900
REPRO_COOLDOWN = 40
REPRO_ENERGY_COST = 30
REPRO_MIN_ENERGY = 55


class Agent:
    def __init__(self, agent_id, sex, x, y, brain=None, rng=None, born_gen=0):
        self.id = agent_id
        self.sex = sex  # 'M' or 'F'
        self.x, self.y = x, y
        self.energy = 70.0
        self.health = MAX_HEALTH
        self.age = 0
        self.wood = 0
        self.repro_cooldown = 0
        self.alive = True
        self.children = 0
        self.born_gen = born_gen
        self.death_cause = None
        self.brain = brain if brain is not None else Brain.random(rng or np.random.default_rng())
        self.last_food_pos = None
        self.last_shelter_pos = None
        self.steps_since_food = 0

    # ---------- الإدراك ----------
    def build_observation(self, world, agents_positions):
        vis = world.vision_grid(self.x, self.y, VISION_RADIUS, agents_positions)

        # تحديث الذاكرة بما هو ظاهر الآن في مجال الرؤية
        food_seen = world.nearest_of(self.x, self.y, VISION_RADIUS, W.FOOD)
        if food_seen:
            self.last_food_pos = food_seen
        shelter_seen = world.nearest_of(self.x, self.y, VISION_RADIUS, W.SHELTER)
        if shelter_seen:
            self.last_shelter_pos = shelter_seen

        day_sin, day_cos = world.time_signals()
        internal = np.array([
            self.energy / MAX_ENERGY,
            self.health / MAX_HEALTH,
            min(self.age / MAX_AGE, 1.0),
            1.0 if self.sex == 'M' else 0.0,
            day_sin, day_cos,
            getattr(self, "_last_temp", 20.0) / 40.0,
            self.repro_cooldown / REPRO_COOLDOWN,
        ], dtype=np.float32)

        def rel(pos):
            if pos is None:
                return (0.0, 0.0)
            dx = ((pos[0] - self.x + world.width // 2) % world.width) - world.width // 2
            dy = ((pos[1] - self.y + world.height // 2) % world.height) - world.height // 2
            return (dx / world.width, dy / world.height)

        fdx, fdy = rel(self.last_food_pos)
        sdx, sdy = rel(self.last_shelter_pos)
        memory = np.array([
            fdx, fdy, sdx, sdy,
            min(self.steps_since_food / 100.0, 1.0),
            min(self.children / 10.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([vis, internal, memory])

    # ---------- الفعل ----------
    def decide(self, obs):
        return self.brain.act(obs)

    def step_body(self, world, rng, is_shelter_here):
        """استهلاك طاقة أساسي + تأثير الحرارة/العمر."""
        temp = world.temperature(rng)
        self._last_temp = temp
        cost = 0.5
        if not world.is_day() and temp < 15 and not is_shelter_here:
            cost += 0.7  # برد الليل بدون مأوى يستنزف أكثر
        self.energy -= cost
        self.age += 1
        self.steps_since_food += 1
        if self.repro_cooldown > 0:
            self.repro_cooldown -= 1

        if self.energy <= 0:
            self.energy = 0
            self.health -= 4  # مجاعة
        if self.age > MAX_AGE:
            self.health -= 0.5  # شيخوخة

        if self.health <= 0:
            self.alive = False
            self.death_cause = "starvation_or_age"

    def apply_action(self, action, world):
        if action < 8:
            dx, dy = MOVE_DELTAS[action]
            nx, ny = world.wrap(self.x + dx, self.y + dy)
            if world.get(nx, ny) not in (W.WATER, W.ROCK):
                self.x, self.y = nx, ny
            else:
                self.energy -= 0.2  # اصطدام بعائق
        elif action == 8:  # EAT
            if world.get(self.x, self.y) == W.FOOD:
                world.set(self.x, self.y, W.EMPTY)
                self.energy = min(MAX_ENERGY, self.energy + 30)
                self.steps_since_food = 0
                return "eat"
        elif action == 9:  # COLLECT
            if world.get(self.x, self.y) == W.TREE:
                self.wood += 1
                world.set(self.x, self.y, W.EMPTY)
                return "collect"
        elif action == 10:  # BUILD
            if self.wood >= 5 and world.get(self.x, self.y) == W.EMPTY:
                world.set(self.x, self.y, W.SHELTER)
                self.wood -= 5
                return "build"
        elif action == 11:  # REPRODUCE -- الفعل الفعلي يُدار من Simulation
            return "reproduce_request"
        return None

    def to_dict(self):
        return {
            "id": self.id, "sex": self.sex, "x": self.x, "y": self.y,
            "energy": self.energy, "health": self.health, "age": self.age,
            "wood": self.wood, "repro_cooldown": self.repro_cooldown,
            "alive": self.alive, "children": self.children,
            "born_gen": self.born_gen, "weights": self.brain.weights.tolist(),
            "last_food_pos": self.last_food_pos, "last_shelter_pos": self.last_shelter_pos,
            "steps_since_food": self.steps_since_food,
        }

    @classmethod
    def from_dict(cls, d):
        a = cls(d["id"], d["sex"], d["x"], d["y"],
                brain=Brain(weights=np.array(d["weights"], dtype=np.float32)))
        a.energy = d["energy"]; a.health = d["health"]; a.age = d["age"]
        a.wood = d["wood"]; a.repro_cooldown = d["repro_cooldown"]
        a.alive = d["alive"]; a.children = d["children"]; a.born_gen = d["born_gen"]
        a.last_food_pos = tuple(d["last_food_pos"]) if d["last_food_pos"] else None
        a.last_shelter_pos = tuple(d["last_shelter_pos"]) if d["last_shelter_pos"] else None
        a.steps_since_food = d["steps_since_food"]
        return a
