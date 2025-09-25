import json
import hashlib


class ParameterHashFunctions:
    def __init__(self):
        pass

    def get_mesh_hash(self, simulation_parameters):
        """
        Create a short hash from mesh-related parameters.
        """
        mesh_parameters = {
            **simulation_parameters.mesh.__dict__,
            "source_centers": simulation_parameters.sources.centers,
            "source_radii": simulation_parameters.sources.radii,
            "polynomial_order": simulation_parameters.solver.polynomial_order,
        }

        encoded = json.dumps(mesh_parameters, sort_keys=True).encode()
        return hashlib.sha1(encoded).hexdigest()[:10]

    def get_simulation_hash(self, path):
        """
        Create a short hash from the simulation parameter file at the given path.
        """
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
