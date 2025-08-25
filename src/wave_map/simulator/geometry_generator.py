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
                 inclusion_semi_major_axis_direction=None):
        self.msh_file = Path(msh_file)
        self.grid_size = grid_size
        self.percent_grid_variation = 0.3
        self.box_size = box_size
        self.source_center = np.array(source_center)
        self.source_radius = source_radius
        self.inclusion_center = np.array(inclusion_center)

        # scaling[0] = semi-major axis, scaling[1] = semi-minor axis (applies to both transverse axes)
        self.inclusion_scaling = np.array(inclusion_scaling)
        self.inclusion_semi_major_axis_direction = np.array(inclusion_semi_major_axis_direction)

    def _initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress output

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
        self.msh_file.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(self.msh_file))
        # gmsh.finalize()

    def _create_domain_box(self):
        return gmsh.model.occ.addBox(0, 0, 0, self.box_size, self.box_size, self.box_size)

    def _add_source_disk(self):
        sx, sy, sz = self.source_center
        return gmsh.model.occ.addDisk(sx, sy, sz, self.source_radius, self.source_radius)

    def _create_scaling_matrix(self):
        """
        Ellipsoid of revolution:
        - Scale along x = semi-major axis (a)
        - Scale along y = semi-minor axis (b)
        - Scale along z = semi-minor axis (b)
        """
        a = self.inclusion_scaling[0]
        b = self.inclusion_scaling[1]
        return np.diag([a, b, b])

    def _create_rotation_matrix_to_align(self):
        """
        Compute rotation that aligns x-axis with inclusion_semi_major_axis_direction.
        """
        v = self.inclusion_semi_major_axis_direction
        v = v / np.linalg.norm(v)  # normalize

        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(v, x_axis):
            return np.eye(3)

        # Axis of rotation = cross product
        axis = np.cross(x_axis, v)
        axis /= np.linalg.norm(axis)

        # Angle = arccos(dot(x_axis, v))
        angle = np.arccos(np.clip(np.dot(x_axis, v), -1.0, 1.0))

        # Rodrigues' rotation formula
        ux, uy, uz = axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        R = np.array([
            [cos_theta + ux**2 * (1 - cos_theta),
             ux * uy * (1 - cos_theta) - uz * sin_theta,
             ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta,
             cos_theta + uy**2 * (1 - cos_theta),
             uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta,
             uz * uy * (1 - cos_theta) + ux * sin_theta,
             cos_theta + uz**2 * (1 - cos_theta)]
        ])
        return R

    def _create_affine_transformation_matrix(self):
        S = self._create_scaling_matrix()
        R = self._create_rotation_matrix_to_align()
        return R @ S  # note: rotate *after* scaling

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

    def generate_ellipsoid_geometry(self):
        logger = getLogger("simlog")
        logger.info("... Generating ellipsoidal inclusion mesh ...")
        self._initialize_gmsh()

        model = gmsh.model
        occ = model.occ
        mesh = model.mesh

        cube = self._create_domain_box()
        source_disk = self._add_source_disk()

        transform_matrix = self._create_affine_transformation_matrix()
        transform_matrix = self._format_transformation_matrix_for_gmsh(transform_matrix)

        # Start from a unit sphere, then scale+rotate+translate
        sphere_tag = occ.addSphere(0, 0, 0, 1.0)
        occ.affineTransform([(3, sphere_tag)], transform_matrix)

        outDimTags, _ = occ.fragment([(3, cube), (3, sphere_tag)], [(2, source_disk)])

        occ.synchronize()

        mesh.setSize(model.getEntities(0), self.grid_size)
        self._label_physical_groups(outDimTags)

        self._finalize_mesh()
        logger.info(f"... Mesh generated: {self.msh_file} ...")
