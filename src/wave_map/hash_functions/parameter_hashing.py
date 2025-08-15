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
                "grid_size": simulation_parameters.mesh.grid_size,
                "box_size": simulation_parameters.mesh.box_size,
                "source_center": simulation_parameters.source.center,
                "source_radius": simulation_parameters.source.radius,
                "inclusion_center": simulation_parameters.mesh.inclusion_center,
                "inclusion_scaling": simulation_parameters.mesh.inclusion_scaling,
                "inclusion_rotation": simulation_parameters.mesh.inclusion_rotation,
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
