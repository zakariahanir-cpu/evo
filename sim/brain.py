"""
شبكة عصبية بسيطة (MLP) تمثل "دماغ" الفرد وفي نفس الوقت جيناته.
الحجم مضبوط ليكون قريب جدًا من 50,000 باراميتر.

المدخلات (498):
  - رؤية محلية محدودة حول الفرد: 4 قنوات × (2*5+1)^2 = 484
  - حالة داخلية (طاقة/صحة/عمر/جنس/وقت/حرارة/تبريد التكاثر) = 8
  - ذاكرة محدودة (اتجاه آخر طعام/مأوى معروف + إحصائيات) = 6

المخرجات (12): 8 اتجاهات حركة + أكل + جمع + بناء + تكاثر
"""
import numpy as np

VISION_RADIUS = 5
INPUT_SIZE = 4 * (2 * VISION_RADIUS + 1) ** 2 + 8 + 6  # = 498
HIDDEN1 = 90
HIDDEN2 = 40
OUTPUT_SIZE = 12

ACTION_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW",
                "EAT", "COLLECT", "BUILD", "REPRODUCE"]

MOVE_DELTAS = {
    0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
    4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1),
}


def param_count():
    return (INPUT_SIZE * HIDDEN1 + HIDDEN1 +
            HIDDEN1 * HIDDEN2 + HIDDEN2 +
            HIDDEN2 * OUTPUT_SIZE + OUTPUT_SIZE)


class Brain:
    """الأوزان مخزّنة كمتجه واحد مسطّح (flat) عشان يسهل التهجين والطفرة."""

    def __init__(self, weights=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        if weights is None:
            self.weights = rng.normal(0, 0.5, size=param_count()).astype(np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
        self._split()

    def _split(self):
        w = self.weights
        i = 0
        s1 = INPUT_SIZE * HIDDEN1
        self.W1 = w[i:i + s1].reshape(INPUT_SIZE, HIDDEN1); i += s1
        self.b1 = w[i:i + HIDDEN1]; i += HIDDEN1
        s2 = HIDDEN1 * HIDDEN2
        self.W2 = w[i:i + s2].reshape(HIDDEN1, HIDDEN2); i += s2
        self.b2 = w[i:i + HIDDEN2]; i += HIDDEN2
        s3 = HIDDEN2 * OUTPUT_SIZE
        self.W3 = w[i:i + s3].reshape(HIDDEN2, OUTPUT_SIZE); i += s3
        self.b3 = w[i:i + OUTPUT_SIZE]; i += OUTPUT_SIZE

    def forward(self, x):
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        return out

    def act(self, x):
        scores = self.forward(x)
        return int(np.argmax(scores))

    @staticmethod
    def crossover(brain_a, brain_b, rng):
        mask = rng.random(param_count()) < 0.5
        child_w = np.where(mask, brain_a.weights, brain_b.weights)
        return child_w

    @staticmethod
    def mutate(weights, rng, rate=0.05, scale=0.3):
        w = weights.copy()
        mask = rng.random(w.shape) < rate
        w[mask] += rng.normal(0, scale, size=mask.sum())
        return w

    @classmethod
    def random(cls, rng):
        return cls(rng=rng)

    @classmethod
    def child_of(cls, parent_a, parent_b, rng):
        w = cls.crossover(parent_a, parent_b, rng)
        w = cls.mutate(w, rng)
        return cls(weights=w)
