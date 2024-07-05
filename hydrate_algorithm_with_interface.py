# Hydrate Composition Calculation Algorithm with Interface




class Structure:

    def __init__(self, deltaMu0, deltaH0, deltaCp0, b, deltaV0, petite_cavite: list, grande_cavite: list) -> None:
        self.deltaMu0 = deltaMu0
        self.deltaH0 = deltaH0
        self.deltaCp0 = deltaCp0
        self.b = b
        self.deltaV0 = deltaV0
        self.cavities = [petite_cavite, grande_cavite]


class Cavity:

    def __init__(self, ray, coord_z, pop_nu) -> None:
        self.r = ray
        self.z = coord_z
        self.nu = pop_nu

class Component:

    def __init__(self, name: str, component_id: int, y, Tc, Pc, epsilon, sigma, a, Vinf, k1, k2, omega) -> None:
        self.name = name
        self.id = component_id
        self.y = y
        self.tc = Tc
        self.pc = Pc
        self.epsilon = epsilon
        self.sigma = sigma
        self.a = a
        self.vinf = Vinf
        self.k1 = k1
        self.k2 = k2
        self.omega = omega

components = []