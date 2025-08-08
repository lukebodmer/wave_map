import gmsh
import numpy as np
from pathlib import Path
from logging import getLogger


class GeometryGenerator:
    def __init__(self,
                 msh_file,
                 grid_size,
                 box_size,
                 source_center,
                 source_radius,
                 inclusion_center,
                 inclusion_scaling=None,
                 inclusion_rotation=None):
        self.msh_file = Path(msh_file)
        self.grid_size = grid_size
        self.percent_grid_variation = 0.3 
        self.box_size = box_size
        self.source_center = np.array(source_center)
        self.source_radius = source_radius
        self.inclusion_center = np.array(inclusion_center)
        self.inclusion_scaling = np.array(inclusion_scaling)
        self.inclusion_rotation = np.array(inclusion_rotation)

    def _initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # don't output

        # Compute min/max element sizes as a percentage variation around grid_size
        variation = self.percent_grid_variation  # 30% +/- around grid_size
        min_size = self.grid_size * (1 - variation)
        max_size = self.grid_size * (1 + variation)

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.clear()
        gmsh.model.add("geometry")

    def _finalize_mesh(self):
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        self.msh_file.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(self.msh_file))
        #gmsh.finalize()

    def _create_domain_box(self):
        x_dim = y_dim = z_dim = self.box_size
        return gmsh.model.occ.addBox(0, 0, 0, x_dim, y_dim, z_dim)

    def _add_source_disk(self):
        sx, sy, sz = self.source_center
        return gmsh.model.occ.addDisk(sx, sy, sz, self.source_radius, self.source_radius)

    def _axes_scaling(self):
        a, b, c = self.inclusion_scaling
        return np.diag([a, b, c])

    def _create_rotation_matrix(self):
        axis = self.inclusion_rotation
        angle = np.linalg.norm(axis)

        # If angle is zero, return identity (no rotation)
        if angle == 0:
            return np.eye(3)

        # Normalize the axis to get a unit vector
        u = axis / angle
        ux, uy, uz = u

        # Precompute trigonometric values
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        # Rodrigues' rotation formula:
        # Computes the rotation matrix for rotating about
        # a unit axis by angle theta
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
        return affine_matrix.flatten().tolist()  # 16 elements, row-major

    def _label_physical_groups(self, entities):
        model = gmsh.model
        for dim, tag in entities:
            if dim == 3:
                model.addPhysicalGroup(3, [tag], tag)

    def _create_affine_transformation_matrix(self):
        S = self._axes_scaling()
        R = self._create_rotation_matrix()
        A = S @ R
        return A
        
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
