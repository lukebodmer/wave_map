class TimeStepSizeCalculator:
    """
    Calculates time step size (dt) based on the CFL condition.
    """

    def __init__(self, max_wave_speed: float, smallest_radii: float, polynomial_order: int):
        self.max_wave_speed = max_wave_speed
        self.smallest_radii = smallest_radii
        self.polynomial_order = polynomial_order

    def calculate_cfl_dt(self) -> float:
        """
        Standard CFL condition:
            dt = smallest_radii / ((2 * polynomial_order + 1) * max_wave_speed)
        """

        cfl_factor = 0.4
        d = self.smallest_radii * 2
        n = self.polynomial_order
        c = self.max_wave_speed
        dt = cfl_factor * (d / (n * n * c))
        #return dt
        return cfl_factor * self.smallest_radii / ((2 * self.polynomial_order + 1) * self.max_wave_speed)
