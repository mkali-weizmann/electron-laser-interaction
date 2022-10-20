from dataclasses import dataclass
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

x = np.linspace(-1, 1, 3)
y = np.linspace(-1, 1, 3)