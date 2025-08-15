import gmsh
import numpy as np
from pathlib import Path
from logging import getLogger


class GmshMeshGenerator:
    def __init__(self, simulation_parameters, mesh_hash):
        self.sim = simulation_parameters

        # Build msh file path from mesh_hash
        meshes_dir = Path(f"data/inputs/meshes/{mesh_hash}")
        meshes_dir.mkdir(parents=True, exist_ok=True)
        self.msh_file = meshes_dir / f"mesh.msh"

        self.grid_size = self.sim.mesh.grid_size
        self.percent_grid_variation = 0.03
        self.box_size = self.sim.mesh.box_size
        self.source_center = np.array(self.sim.source.center)
        self.source_radius = self.sim.source.radius
        self.inclusion_center = np.array(self.sim.mesh.inclusion_center)
        self.inclusion_scaling = np.array(self.sim.mesh.inclusion_scaling)
        self.inclusion_rotation = np.array(self.sim.mesh.inclusion_rotation)

        self.smallest_radii = None

    def _initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        variation = self.percent_grid_variation
        min_size = self.grid_size * (1 - variation)
        max_size = self.grid_size * (1 + variation)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        gmsh.clear()
        gmsh.model.add("geometry")

    def _finalize_mesh(self):
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(self.msh_file))
        # gmsh.finalize()  # Keep open if caller wants more queries

    def _create_domain_box(self):
        return gmsh.model.occ.addBox(0, 0, 0, self.box_size, self.box_size, self.box_size)

    def _add_source_disk(self):
        sx, sy, sz = self.source_center
        return gmsh.model.occ.addDisk(sx, sy, sz, self.source_radius, self.source_radius)

    def _axes_scaling(self):
        a, b, c = self.inclusion_scaling
        return np.diag([a, b, c])

    def _create_rotation_matrix(self):
        axis = self.inclusion_rotation
        angle = np.linalg.norm(axis)

        if angle == 0:
            return np.eye(3)

        u = axis / angle
        ux, uy, uz = u
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        return np.array([
            [cos_theta + ux**2 * one_minus_cos,
             ux * uy * one_minus_cos - uz * sin_theta,
             ux * uz * one_minus_cos + uy * sin_theta],
            [uy * ux * one_minus_cos + uz * sin_theta,
             cos_theta + uy**2 * one_minus_cos,
             uy * uz * one_minus_cos - ux * sin_theta],
            [uz * ux * one_minus_cos - uy * sin_theta,
             uz * uy * one_minus_cos + ux * sin_theta,
             cos_theta + uz**2 * one_minus_cos]
        ])

    def _format_transformation_matrix_for_gmsh(self, A):
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = A
        affine_matrix[:3, 3] = self.inclusion_center
        return affine_matrix.flatten().tolist()

    def _label_physical_groups(self, entities):
        model = gmsh.model
        for dim, tag in entities:
            if dim == 3:
                model.addPhysicalGroup(3, [tag], tag)

    def _create_affine_transformation_matrix(self):
        S = self._axes_scaling()
        R = self._create_rotation_matrix()
        return S @ R

    def generate_ellipsoid_geometry(self):
        logger = getLogger("simlog")
        logger.info("... Generating ellipsoidal inclusion mesh ...")
        self._initialize_gmsh()

        # some API shortcuts
        model = gmsh.model
        occ = model.occ
        mesh = model.mesh

        # create domain and source
        cube = self._create_domain_box()
        source_disk = self._add_source_disk()

        # generate affine matrix to transform sphere into ellipsoid
        transform_matrix = self._create_affine_transformation_matrix()
        transform_matrix = self._format_transformation_matrix_for_gmsh(transform_matrix)

        # transform sphere into ellipsoid
        sphere_tag = occ.addSphere(0, 0, 0, 1.0)
        occ.affineTransform([(3, sphere_tag)], transform_matrix)

        # combine all meshes into single mesh
        outDimTags, _ = occ.fragment([(3, cube), (3, sphere_tag)], [(2, source_disk)])

        # create gmsh mesh
        occ.synchronize()

        mesh.setSize(model.getEntities(0), self.grid_size)
        self._label_physical_groups(outDimTags)

        self._finalize_mesh()
        logger.info(f"... Mesh generated: {self.msh_file} ...")

    def get_smallest_radii(self):
        _, eleTags, _ = gmsh.model.mesh.getElements(dim=3)
        radii = gmsh.model.mesh.getElementQualities(eleTags[0], "innerRadius")
        self.smallest_radii = np.min(radii)
        return self.smallest_radii
