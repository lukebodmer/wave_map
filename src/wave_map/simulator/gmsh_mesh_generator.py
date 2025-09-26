import gmsh
import numpy as np
from pathlib import Path
from logging import getLogger

BATCH_DATA_DIR = "data/simulation_batch_data"


class GmshMeshGenerator:
    def __init__(self, simulation_parameters, mesh_hash, batch_name):
        # Get mesh path
        self.base_output_path = Path(f"{BATCH_DATA_DIR}/{batch_name}")
        self.mesh_base_output_path = self.base_output_path / "meshes"

        # Create msh file path from mesh_hash
        meshes_dir = self.mesh_base_output_path / mesh_hash
        meshes_dir.mkdir(parents=True, exist_ok=True)
        self.msh_file = meshes_dir / "mesh.msh"

        # Get simulation parameters
        self.sim_params = simulation_parameters
        self.grid_size = self.sim_params.mesh.grid_size
        self.percent_grid_variation = 0.03
        self.box_size = self.sim_params.mesh.box_size
        self.source_centers = np.array(self.sim_params.sources.centers)
        self.source_radii = self.sim_params.sources.radii
        self.inclusion_center = np.array(self.sim_params.mesh.inclusion_center)
        self.inclusion_scaling = np.array(self.sim_params.mesh.inclusion_scaling)
        self.inclusion_semi_major_axis_direction = np.array(self.sim_params.mesh.inclusion_semi_major_axis_direction)
        self.number_of_cubes = self.sim_params.mesh.number_of_cubes
        self.cube_centers = self.sim_params.mesh.cube_centers
        self.cube_widths = self.sim_params.mesh.cube_widths

        self.smallest_radii = None

    def _initialize_gmsh(self):
        if not gmsh.isInitialized():
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
        domain_tag = gmsh.model.occ.addBox(0, 0, 0, self.box_size, self.box_size, self.box_size)
        return domain_tag

    #def _add_source_disk(self):
    #    sx, sy, sz = self.source_center
    #    return gmsh.model.occ.addDisk(sx, sy, sz, self.source_radius, self.source_radius)

    def _add_source_disks(self):
        """
        Create one gmsh disk for each source center / radius,
        oriented in the correct boundary plane.
        Returns a list of (2, tag) entities for occ.fragment.
        """
        occ = gmsh.model.occ
        disks = []
        tol = 1e-8

        for j, ((sx, sy, sz), r) in enumerate(zip(self.source_centers, self.source_radii)):
            # default: disk lies in XY plane at (sx, sy, sz)
            tag = occ.addDisk(float(sx), float(sy), float(sz), float(r), float(r))

            # figure out which boundary face it's on
            if abs(sx - 0.0) < tol or abs(sx - self.box_size) < tol:
                # plane is yz at x=const → rotate disk (originally in XY plane) around Y-axis by 90°
                occ.rotate([(2, tag)], sx, sy, sz, 0, 1, 0, np.pi / 2)

            elif abs(sy - 0.0) < tol or abs(sy - self.box_size) < tol:
                # plane is xz at y=const → rotate disk around X-axis by -90°
                occ.rotate([(2, tag)], sx, sy, sz, 1, 0, 0, -np.pi / 2)

            elif abs(sz - 0.0) < tol or abs(sz - self.box_size) < tol:
                # plane is xy at z=const → no rotation needed
                pass

            else:
                raise ValueError(f"Source center {(sx, sy, sz)} is not on a boundary plane.")

            disks.append((2, tag))

        return disks

    def _axes_scaling(self):
        a, b, c = self.inclusion_scaling
        return np.diag([a, b, c])

    def _create_rotation_matrix(self):
        """
        Create rotation matrix that aligns the x-axis [1,0,0] with the
        inclusion_semi_major_axis_direction vector.
        """
        # normalize target direction
        v = self.inclusion_semi_major_axis_direction
        if np.linalg.norm(v) == 0:
            return np.eye(3)  # no rotation

        v = v / np.linalg.norm(v)
        x_axis = np.array([1.0, 0.0, 0.0])

        # compute rotation axis and angle
        axis = np.cross(x_axis, v)
        angle = np.linalg.norm(axis)

        if angle == 0:
            return np.eye(3)  # already aligned

        axis = axis / angle  # normalize rotation axis
        ux, uy, uz = axis
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1 - cos_theta

        return np.array([
            [cos_theta + ux**2 * one_minus_cos,
             ux*uy*one_minus_cos - uz*sin_theta,
             ux*uz*one_minus_cos + uy*sin_theta],
            [uy*ux*one_minus_cos + uz*sin_theta,
             cos_theta + uy**2*one_minus_cos,
             uy*uz*one_minus_cos - ux*sin_theta],
            [uz*ux*one_minus_cos - uy*sin_theta,
             uz*uy*one_minus_cos + ux*sin_theta,
             cos_theta + uz**2*one_minus_cos]
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
 
    def get_smallest_radii(self):
        _, eleTags, _ = gmsh.model.mesh.getElements(dim=3)
        radii = gmsh.model.mesh.getElementQualities(eleTags[0], "innerRadius")
        self.smallest_radii = np.min(radii)
        return self.smallest_radii

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
        #source_disk = self._add_source_disk()
        source_disks = self._add_source_disks()

        # generate affine matrix to transform sphere into ellipsoid
        transform_matrix = self._create_affine_transformation_matrix()
        transform_matrix = self._format_transformation_matrix_for_gmsh(transform_matrix)

        # transform sphere into ellipsoid
        sphere_tag = occ.addSphere(0, 0, 0, 1.0)
        occ.affineTransform([(3, sphere_tag)], transform_matrix)

        # combine all meshes into single mesh
        #outDimTags, _ = occ.fragment([(3, cube), (3, sphere_tag)], [(2, source_disk)])
        outDimTags, _ = occ.fragment([(3, cube), (3, sphere_tag)], source_disks)

        # create gmsh mesh
        occ.synchronize()

        mesh.setSize(model.getEntities(0), self.grid_size)
        self._label_physical_groups(outDimTags)
        self._finalize_mesh()
        logger.info(f"... Mesh generated: {self.msh_file} ...")

    def generate_multi_cube_geometry(self):
        """
        Generate geometry with multiple cubes embedded inside the domain box.
        Uses self.number_of_cubes, self.cube_centers, and self.cube_widths.
        """
        logger = getLogger("simlog")
        logger.info("... Generating multi-cube inclusion mesh ...")
        self._initialize_gmsh()

        model = gmsh.model
        occ = model.occ
        mesh = model.mesh

        # Create the outer domain box
        domain_box = self._create_domain_box()

        # Create all cube inclusions
        cube_tags = []
        for i in range(self.number_of_cubes):
            cx, cy, cz = self.cube_centers[i]
            w = self.cube_widths[i]

            # gmsh.addBox wants corner coordinates, so shift from center
            x0 = cx - w / 2
            y0 = cy - w / 2
            z0 = cz - w / 2
            tag = occ.addBox(x0, y0, z0, w, w, w)
            cube_tags.append((3, tag))

        # add source disks
        source_disks = self._add_source_disks()

        # Combine domain and inclusions
        #outDimTags, _ = occ.fragment([(3, domain_box)], cube_tags)
        # fragment domain + cubes with disks
        solids = [(3, domain_box)] + cube_tags
        outDimTags, _ = occ.fragment(solids, source_disks)

        # Synchronize CAD kernel with Gmsh model
        occ.synchronize()

        # --- Assign physical groups using outDimTags ---
        # We'll classify volumes (dim=3) as OuterDomain or CubeN
        cube_centers = np.array(self.cube_centers)
        tol = 1e-6  # small tolerance for bounding box check

        for dim, tag in outDimTags:
            if dim == 3:
                # Get the bounding box of this volume
                xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(dim, tag)
                center = np.array([(xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2])

                # Check if this volume matches any cube center
                matched = False
                for i, cc in enumerate(cube_centers):
                    if np.all(np.abs(center - cc) < tol):
                        phys = model.addPhysicalGroup(3, [tag])
                        model.setPhysicalName(3, phys, f"Cube{i}")
                        matched = True
                        break
                if not matched:
                    # Otherwise, it's the outer domain
                    phys = model.addPhysicalGroup(3, [tag])
                    model.setPhysicalName(3, phys, "BackgroundMaterial")

            elif dim == 2:
                # Surfaces (source disks)
                phys = model.addPhysicalGroup(2, [tag])
                model.setPhysicalName(2, phys, f"Source{tag}")

        # Mesh sizing
        mesh.setSize(model.getEntities(0), self.grid_size)

        # Label physical groups for all 3D entities
        #self._label_physical_groups(outDimTags)

        # Finalize mesh
        self._finalize_mesh()
        logger.info(f"... Multi-cube mesh generated: {self.msh_file} ...")
