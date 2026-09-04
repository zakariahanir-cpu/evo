"""
عالم المحاكاة: شبكة (grid) تلتف على نفسها كأنها سطح كوكب (toroidal world).
تحتوي على: يابسة فارغة، طعام، أشجار (خشب)، ماء (عائق)، عوائق صخرية، مأوى مبني.
فيها دورة ليل/نهار وحرارة متغيرة تؤثر على استهلاك الطاقة.
"""
import numpy as np

EMPTY = 0
FOOD = 1
TREE = 2
WATER = 3
ROCK = 4
SHELTER = 5

CELL_NAMES = {
    EMPTY: "فارغ", FOOD: "طعام", TREE: "شجرة",
    WATER: "ماء", ROCK: "صخرة", SHELTER: "مأوى",
}

# خلايا لا يمكن للفرد الدخول فيها فعليًا
BLOCKING = {WATER, ROCK}

DAY_LENGTH = 100  # عدد النبضات (ticks) في اليوم الواحد


class World:
    def __init__(self, width=50, height=50, seed=None):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)
        self.grid = np.zeros((height, width), dtype=np.int8)
        self._scatter(rng, WATER, 0.06, cluster=True)
        self._scatter(rng, ROCK, 0.04, cluster=False)
        self._scatter(rng, TREE, 0.08, cluster=True)
        self._scatter(rng, FOOD, 0.10, cluster=False)
        self.tick = 0

    def _scatter(self, rng, cell_type, density, cluster=False):
        n = int(self.width * self.height * density)
        if not cluster:
            for _ in range(n):
                y, x = rng.integers(0, self.height), rng.integers(0, self.width)
                if self.grid[y, x] == EMPTY:
                    self.grid[y, x] = cell_type
        else:
            clusters = max(1, n // 6)
            for _ in range(clusters):
                cy, cx = rng.integers(0, self.height), rng.integers(0, self.width)
                for _ in range(6):
                    y = (cy + rng.integers(-2, 3)) % self.height
                    x = (cx + rng.integers(-2, 3)) % self.width
                    if self.grid[y, x] == EMPTY:
                        self.grid[y, x] = cell_type

    def wrap(self, x, y):
        return x % self.width, y % self.height

    def get(self, x, y):
        x, y = self.wrap(x, y)
        return int(self.grid[y, x])

    def set(self, x, y, val):
        x, y = self.wrap(x, y)
        self.grid[y, x] = val

    def is_day(self):
        phase = (self.tick % DAY_LENGTH) / DAY_LENGTH
        return np.sin(2 * np.pi * phase) > 0

    def time_signals(self):
        phase = (self.tick % DAY_LENGTH) / DAY_LENGTH
        return np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

    def temperature(self, rng):
        s, _ = self.time_signals()
        base = 20 + 10 * s
        noise = rng.normal(0, 1.5)
        return base + noise

    def step(self, rng):
        self.tick += 1
        # تجدد طبيعي بطيء للطعام والأشجار
        if rng.random() < 0.5:
            n_new = rng.integers(1, 4)
            for _ in range(n_new):
                y, x = rng.integers(0, self.height), rng.integers(0, self.width)
                if self.grid[y, x] == EMPTY and rng.random() < 0.4:
                    self.grid[y, x] = FOOD
                elif self.grid[y, x] == EMPTY and rng.random() < 0.1:
                    self.grid[y, x] = TREE

    def vision_grid(self, x, y, radius, agents_positions):
        """يرجع شبكة رؤية محلية فقط (مش الخريطة كاملة) حول الفرد."""
        size = 2 * radius + 1
        food_ch = np.zeros((size, size), dtype=np.float32)
        tree_ch = np.zeros((size, size), dtype=np.float32)
        block_ch = np.zeros((size, size), dtype=np.float32)
        agent_ch = np.zeros((size, size), dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cx, cy = self.wrap(x + dx, y + dy)
                cell = self.grid[cy, cx]
                iy, ix = dy + radius, dx + radius
                if cell == FOOD:
                    food_ch[iy, ix] = 1.0
                elif cell == TREE:
                    tree_ch[iy, ix] = 1.0
                elif cell in BLOCKING:
                    block_ch[iy, ix] = 1.0
                if (cx, cy) in agents_positions:
                    agent_ch[iy, ix] = 1.0
        return np.stack([food_ch, tree_ch, block_ch, agent_ch]).flatten()

    def nearest_of(self, x, y, radius, cell_type):
        """أقرب خلية من نوع معين ضمن مجال الرؤية، أو None."""
        best = None
        best_d = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cx, cy = self.wrap(x + dx, y + dy)
                if self.grid[cy, cx] == cell_type:
                    d = dx * dx + dy * dy
                    if best_d is None or d < best_d:
                        best_d = d
                        best = (cx, cy)
        return best

    def to_dict(self):
        return {"width": self.width, "height": self.height,
                "grid": self.grid.tolist(), "tick": self.tick}

    @classmethod
    def from_dict(cls, d):
        w = cls.__new__(cls)
        w.width = d["width"]
        w.height = d["height"]
        w.grid = np.array(d["grid"], dtype=np.int8)
        w.tick = d["tick"]
        return w
