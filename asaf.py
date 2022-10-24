from dataclasses import asdict, dataclass
from typing import List



class Transformer:
    def transform(self, state: WhateverState) -> WhateverState:
        raise NotImplementedError()


class AsafTransofmer(Transformer):
    pass


@dataclass
class Activation:
    input: WhateverState
    output: WhateverState
    transformer: Transformer


class Microscope:
    def __init__(self, transformers: List[Transformer]):
        self.transformers = transformers
        self.activations: List[Activation] = []

    def take_a_picture(self, input: WhateverState) -> WhateverState:
        for transformer in self.transformers:
            output = transformer.transform(input)
            self.activations.append(Activation(input, output, transformer))
            input = output
        return input



class WhateverState:
    def __init__(self ,psi, coordinates):
        self.psi = psi
        self.coordinates = coordinates


class ATransformer(Transformer):
    def __init__(self, a):
        self.a = a

    def transform(self, state: WhateverState) -> WhateverState:
        state.psi += self.a
        return state


class ABTransformer(Transformer):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def transform(self, state: WhateverState) -> WhateverState:
        state.psi *= (self.a + self.b)
        return state
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
# %%
def f(x, k, A, c):
    return np.exp(1j*A*np.cos(-k*x + c))

def f_sin(x, k, A, c):
    return np.exp(1j*A*np.sin(-k*x+np.pi/2 + c))

def f_approx(x: np.ndarray, k, A, c):
    qs = np.arange(-50, 50, 1)
    return sum([jv(q, A) * np.exp(1j*(q*(-k*x-np.pi/2+c))) for q in qs])

x = np.linspace(-10, 10, 100)
k = 0.7
A = 0.1
c=0
f_values = f(x, k, A, c)
f_sin_values = f_sin(x, k, A, c)
f_approx_values = f_approx(x, k, A, c)

plt.plot(x, np.real(f_values), 'r--', linewidth=3)
plt.plot(x, np.real(f_approx_values), 'b--')
plt.plot(x, np.real(f_approx_values), 'g.')
plt.show()
