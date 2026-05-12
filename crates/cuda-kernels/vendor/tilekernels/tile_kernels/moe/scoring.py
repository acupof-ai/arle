from enum import IntEnum
from tilelang import language as T


class ScoringFunc(IntEnum):
    SIGMOID = 0
    SQRTSOFTPLUS = 1
    SOFTMAX = 2
    IDENTITY = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, label: str):
        try:
            return cls[label.upper()]
        except KeyError:
            raise ValueError(f'{label} is not a valid {cls.__name__}')


@T.macro
def softplus(x: T.Ref):
    threshold = 20.0
    return T.if_then_else(x > threshold, x, T.log1p(T.exp(x)))

