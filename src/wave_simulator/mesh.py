import numpy as np
import gmsh
from logging import getLogger
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.geometry_generator import GeometryGenerator


class Mesh3d:
    def __init__(self,
                 finite_element: LagrangeElement,
                 msh_file=None,
                 grid_size=None,
                 box_size=None,
                 source_center=None,
                 source_radius=None,
                 outer_density=None,
                 outer_speed=None,
                 inclusion_density=None,
                 inclusion_speed=None,
                 inclusion_center=None,
                 inclusion_scaling=None,
                 inclusion_rotation=None
                 ):
        if None not in (msh_file,
                        grid_size,
                        box_size,
                        source_center,
                        source_radius,
                        outer_density,
                        outer_speed,
                        inclusion_density,
                        inclusion_speed,
                        inclusion_center,
                        inclusion_scaling,
                        inclusion_rotation,
                        ):
            self.reference_element = finite_element
            self.reference_element_operators = ReferenceElementOperators(self.reference_element)
            self.dim = self.reference_element.d
            self.n = self.reference_element.n  # polynomial order

            self.msh_file = msh_file
            self.grid_size = grid_size
            self.box_size = box_size
            self.source_center = source_center
            self.source_radius = source_radius
            self.outer_density = outer_density
            self.outer_speed = outer_speed
            self.inclusion_density = inclusion_density
            self.inclusion_speed = inclusion_speed
            self.inclusion_center= inclusion_center
            self.inclusion_scaling= inclusion_scaling
            self.inclusion_rotation= inclusion_rotation
        else:
            raise ValueError("Invalid Mesh3d initialization: must provide all geometric parameters.")

        if msh_file.exists():
            self.initialize_gmsh()
        else:
            self._generate_geometry()

       # self.num_vertices = 0
       # self.num_cells= 0
       # self.vertex_coordinates = []
       # self.x_vertex = [] # vertex x coordinates
       # self.y_vertex = [] # vertex y coordinates
       # self.z_vertex = [] # vertex z coordinates
       # self.x = []  # nodal x coordinates
       # self.y = []  # nodal y coordinates
       # self.z = []  # nodal z coordinates
       # self.nx = None
       # self.ny = None
       # self.nz = None
       # self.edge_vertices = []
       # self.cell_to_vertices = []
       # self.cell_to_cells = []
       # self.cell_to_faces = []
       # reference to physical mapping coefficients
       # self.drdx = None
       # self.drdy = None
       # self.drdz = None
       # self.dsdx = None
       # self.dsdy = None
       # self.dsdz = None
       # self.dtdx = None
       # self.dtdy = None
       # self.dtdz = None
       # self.jacobians = {}
       # #self.determinants = {}
       
#        self.initialize_gmsh()
        self._extract_mesh_info()
        self._get_material_info()
        self._get_smallest_diameter()
        self._build_connectivityMatricies()
        #self._compute_gmsh_jacobians()
        self._get_mapped_nodal_cordinates()
        self._compute_mapping_coefficients()
        self._compute_normals_at_face_nodes()
        self._compute_face_node_mappings()
        self._find_boundary_nodes()
        self._compute_surface_to_volume_jacobian()
        self.log_info()

    def initialize_gmsh(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        logger = getLogger("simlog")
        logger.info(f"... Found mesh file {self.msh_file}")
        logger.info("... Processing mesh file  ...")
        gmsh.open(str(self.msh_file))

    def _generate_geometry(self):
        logger = getLogger("simlog")
        logger.info("... Mesh not found. Generating new mesh ...")

        geom = GeometryGenerator(
            msh_file=self.msh_file,
            grid_size=self.grid_size,
            box_size=self.box_size,
            source_center=self.source_center,
            source_radius=self.source_radius,
            inclusion_center=self.inclusion_center,
            inclusion_scaling=self.inclusion_scaling,
            inclusion_rotation=self.inclusion_rotation,
        )

        geom.generate_ellipsoid_geometry()
#
#    def _generate_geometry(self):
#        logger = getLogger("simlog")
#        logger.info("... Mesh not found. Generating new mesh ...")
#
#        gmsh.initialize()
#        gmsh.option.setNumber("General.Terminal", 0);
#        gmsh.clear()
#
#        # Some abbreviations
#        model = gmsh.model
#        mesh = model.mesh
# 
#        # Extract geometry parameters
#        x_dim = self.box_size
#        y_dim = self.box_size
#        z_dim = self.box_size
#    
#        # Source geometry
#        source_x, source_y, source_z = self.source_center
#        source_radius = self.source_radius
#    
#        # Inclusion geometry
#        #inclusion_center = (x_dim / 2, y_dim / 2, z_dim / 2)
#        inclusion_center = self.inclusion_center
#        inclusion_radius = self.inclusion_radius
#        
#        main_cell_size = self.grid_size
#        
#        # Create outer box
#        cube = model.occ.addBox(0, 0, 0, x_dim, y_dim, z_dim)
#    
#        # Create central inclusion
#        inclusion = model.occ.addSphere(
#            inclusion_center[0],
#            inclusion_center[1],
#            inclusion_center[2],
#            inclusion_radius
#        )
#
#        # Create source disk
#        source_disk = model.occ.addDisk(source_x, source_y, source_z, source_radius, source_radius)
#    
#        # Perform boolean fragment
#        outDimTags, _ = model.occ.fragment(
#            [(3, cube), (3, inclusion), (2, source_disk)],
#            []
#        )
#    
#        model.occ.synchronize()
#    
#        # Set mesh size
#        mesh.setSize(model.getEntities(0), main_cell_size)
#    
#        # Add physical groups for volumes only
#        volume_count = 0
#        for dim, tag in outDimTags:
#            if dim == 3:
#                volume_count += 1
#                model.addPhysicalGroup(3, [tag], tag=volume_count)
#    
#        # Optimize mesh
#        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
#    
#        # Generate and save mesh
#        model.mesh.generate(3)
#
#        self.msh_file.parent.mkdir(parents=True, exist_ok=True)
#        gmsh.write(str(self.msh_file))
#
#        logger.info(f"... Mesh generated: {self.msh_file} ...")

    def _extract_mesh_info(self):
        """ Get information from Gmsh file """
        # get vertex information 
        ntags, coords, _ = gmsh.model.mesh.getNodes(4)
        self.num_vertices= len(ntags)
        self.vertex_coordinates = coords.reshape(-1, 3)
        self.x_vertex = self.vertex_coordinates[:, 0]
        self.y_vertex = self.vertex_coordinates[:, 1]
        self.z_vertex = self.vertex_coordinates[:, 2]

        # get cell information
        # get all the nodes from tetrahedrons (elementType = 4)
        node_tags, _, _ = gmsh.model.mesh.getNodesByElementType(4) 
        self.num_cells = int(len(node_tags)/4) 
        self.cell_to_vertices = node_tags.reshape(-1, 4).astype(int) - 1

    def _get_material_info(self):
        # Material Speed List (m/s)
        # bone = 2600 (Thomas Riis 2021)
        # EcoFlex00-10 = 974 (Cafarelli 2016)
        # Polyurethane rubber = 1398
        # Material Density List (kg/m^3)
        # bone = 2000 (Hamed Abdi 2024)
        # EcoFlex = 1063
        # Polyurethane rubber = 1016

        # input order: gray matter, bone, white matter 
        #speed = [1398 , 2600, 974] # m/s
        #density = [1016, 2000, 1063] # kg/m^3
        #speed = [3.0 , 1.5] # m/s
        #density = [8.0, 1.0] # kg/m^3
        speed = [self.inclusion_speed, self.outer_speed]  # physical group tags 1, 2
        density = [self.inclusion_density, self.outer_density]

        dim = 3
        physical_groups = gmsh.model.getPhysicalGroups(dim)
        self.speed = np.ones((self.reference_element.nodes_per_cell, self.num_cells)) # m/s 
        self.density = np.ones((self.reference_element.nodes_per_cell, self.num_cells)) # kg/m^3

        # loop over all physical_groups
        for i, group in enumerate(physical_groups):
            dim = group[0]
            tag = group[1]
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, entity)
                offset, _ = gmsh.model.mesh.getElementsByType(4)
                self.speed[:, np.array(elemTags)-offset[0]] = speed[i]
                self.density[:, np.array(elemTags)-offset[0]] = density[i]

    def _get_smallest_diameter(self):
        _, eleTags , _ = gmsh.model.mesh.getElements(dim=3)
        radii = gmsh.model.mesh.getElementQualities(eleTags[0], "innerRadius")
        self.smallest_diameter = np.min(radii) * 2

    def _build_connectivityMatricies(self):
        """tetrahedral face connect algorithm from Toby Isaac"""
        num_faces = 4
        K = self.num_cells
        CtoV = self.cell_to_vertices
        num_vertices = self.num_vertices 
        
        # create list of all faces
        face_vertices = np.vstack((CtoV[:, [0, 1, 2]],
                                   CtoV[:, [0, 1, 3]],
                                   CtoV[:, [1, 2, 3]],
                                   CtoV[:, [0, 2, 3]]))

        # sort each row from low to high for hash algorithm
        face_vertices = np.sort(face_vertices, axis=1)
         
        # unique hash for each set of three faces by their vertex numbers
        face_hashes = face_vertices[:, 0] * num_vertices * num_vertices  + \
                     face_vertices[:, 1] * num_vertices + \
                     face_vertices[:, 2] + 1

        # vertex id from 1 - num_faces* num_cells
        vertex_ids = np.arange(0, num_faces*K)
       
        # set up default cell to cell and cell to faces connectivity
        CtoC = np.tile(np.arange(K)[:, np.newaxis], num_faces)
        CtoF = np.tile(np.arange(num_faces), (K,1))

        # build a master matrix (mappingTable) that we will solve by 
        # sorting by one column to create the connectivity matricies
        mapping_table = np.column_stack((face_hashes,
                                        vertex_ids,
                                        np.ravel(CtoC, order='F'),
                                        np.ravel(CtoF, order='F')))
        
        # Now we sort by global face number.
        sorted_map_table= np.array(sorted(mapping_table, key=lambda x: (x[0], x[1])))
        
        # find matches in the sorted face list
        matches = np.where(sorted_map_table[:-1, 0] == sorted_map_table[1:, 0])[0]
        
        # make links reflexive
        match_l = np.vstack((sorted_map_table[matches], sorted_map_table[matches + 1]))
        match_r = np.vstack((sorted_map_table[matches + 1], sorted_map_table[matches]))
        
        # insert matches
        CtoC_tmp = np.ravel(CtoC, order='F')
        CtoF_tmp = np.ravel(CtoF, order='F')
        CtoC_tmp[match_l[:, 1]] = match_r[:, 2]
        CtoF_tmp[match_l[:, 1]] = match_r[:, 3]
        
        CtoC = CtoC_tmp.reshape(CtoC.shape, order='F')
        CtoF = CtoF_tmp.reshape(CtoF.shape, order='F')

        self.cell_to_cells = CtoC
        self.cell_to_faces = CtoF

    def _compute_gmsh_jacobians(self):
        """ calculate the jacobian of the mapping of each cell """
        # get local coordinates of the verticies in the
        # reference tetrahedron
        name, dim, order, numNodes, localCoords, _ = gmsh.model.mesh.getElementProperties(4)
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(4, localCoords)
        self.gmsh_jacobians = jacobians.reshape(-1, 3, 3)
        self.gmsh_determinants = determinants
        
    def _get_mapped_nodal_cordinates(self):
        """ returns x, y, and z arrays of coordinates of nodes from EToV and VX, VY, VZ, arrays"""
        CtoV = self.cell_to_vertices
        vx = self.x_vertex
        vy = self.y_vertex
        vz = self.z_vertex
        r = self.reference_element.r
        s = self.reference_element.s
        t = self.reference_element.t

        # extract vertex numbers from elements
        va = CtoV[:, 0].T
        vb = CtoV[:, 1].T
        vc = CtoV[:, 2].T
        vd = CtoV[:, 3].T
        
        vx = vx.reshape(-1, 1)
        vy = vy.reshape(-1, 1)
        vz = vz.reshape(-1, 1)
        
        # map r, s, t from standard tetrahedron to x, y, z coordinates for each element
        self.x = (0.5 * (-(1 + r + s + t) * vx[va] + (1 + r) * vx[vb] + (1 + s) * vx[vc] + (1 + t) * vx[vd])).T
        self.y = (0.5 * (-(1 + r + s + t) * vy[va] + (1 + r) * vy[vb] + (1 + s) * vy[vc] + (1 + t) * vy[vd])).T
        self.z = (0.5 * (-(1 + r + s + t) * vz[va] + (1 + r) * vz[vb] + (1 + s) * vz[vc] + (1 + t) * vz[vd])).T

       
    def _compute_mapping_coefficients(self):
        """Compute the metric elements for the local mappings of the elements"""
        Dr = self.reference_element_operators.r_differentiation_matrix
        Ds = self.reference_element_operators.s_differentiation_matrix
        Dt = self.reference_element_operators.t_differentiation_matrix
        x = self.x
        y = self.y
        z = self.z

        # find jacobian of mapping
        xr = np.dot(Dr, x)
        xs = np.dot(Ds, x)
        xt = np.dot(Dt, x)
        yr = np.dot(Dr, y)
        ys = np.dot(Ds, y)
        yt = np.dot(Dt, y)
        zr = np.dot(Dr, z)
        zs = np.dot(Ds, z)
        zt = np.dot(Dt, z)

        J = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt)
        self.jacobians = J

        # compute the metric constants
        self.drdx = (ys * zt - zs * yt) / J
        self.drdy = -(xs * zt - zs * xt) / J
        self.drdz = (xs * yt - ys * xt) / J
        self.dsdx = -(yr * zt - zr * yt) / J
        self.dsdy = (xr * zt - zr * xt) / J
        self.dsdz = -(xr * yt - yr * xt) / J
        self.dtdx = (yr * zs - zr * ys) / J
        self.dtdy = -(xr * zs - zr * xs) / J
        self.dtdz = (xr * ys - yr * xs) / J

    def _compute_normals_at_face_nodes(self):
        """compute outward pointing normals at elements faces as well as surface Jacobians"""
        Nfp = self.reference_element.nodes_per_face
        K = self.num_cells
        num_faces = self.reference_element.num_faces
        
        face_node_indices = self.reference_element.face_node_indices

        # interpolate geometric factors to face nodes
        face_drdx = self.drdx[face_node_indices, :]
        face_dsdx = self.dsdx[face_node_indices, :]
        face_dtdx = self.dtdx[face_node_indices, :]
        face_drdy = self.drdy[face_node_indices, :]
        face_dsdy = self.dsdy[face_node_indices, :]
        face_dtdy = self.dtdy[face_node_indices, :]
        face_drdz = self.drdz[face_node_indices, :]
        face_dsdz = self.dsdz[face_node_indices, :]
        face_dtdz = self.dtdz[face_node_indices, :]

        # build normal vectors
        nx = np.zeros((num_faces * Nfp, K))
        ny = np.zeros((num_faces * Nfp, K))
        nz = np.zeros((num_faces * Nfp, K))

        # create vectors of indices of each face
        face_0_indices = np.arange(0, Nfp)
        face_1_indices = np.arange(Nfp, 2 * Nfp)
        face_2_indices = np.arange(2 * Nfp, 3 * Nfp)
        face_3_indices = np.arange(3 * Nfp, 4 * Nfp)

        # face 0: t = -1 → Normal in -t direction
        nx[face_0_indices, :] = -face_dtdx[face_0_indices, :]
        ny[face_0_indices, :] = -face_dtdy[face_0_indices, :]
        nz[face_0_indices, :] = -face_dtdz[face_0_indices, :]

        # face 1: s = -1 → Normal in -s direction
        nx[face_1_indices, :] = -face_dsdx[face_1_indices, :]
        ny[face_1_indices, :] = -face_dsdy[face_1_indices, :]
        nz[face_1_indices, :] = -face_dsdz[face_1_indices, :]

        # face 2: r + s + t = -1 → Normal is the gradient of (r + s + t)
        nx[face_2_indices, :] = face_drdx[face_2_indices, :] + face_dsdx[face_2_indices, :] + face_dtdx[face_2_indices, :]
        ny[face_2_indices, :] = face_drdy[face_2_indices, :] + face_dsdy[face_2_indices, :] + face_dtdy[face_2_indices, :]
        nz[face_2_indices, :] = face_drdz[face_2_indices, :] + face_dsdz[face_2_indices, :] + face_dtdz[face_2_indices, :]

        # face 3: r = -1 → Normal in -r direction
        nx[face_3_indices, :] = -face_drdx[face_3_indices, :]
        ny[face_3_indices, :] = -face_drdy[face_3_indices, :]
        nz[face_3_indices, :] = -face_drdz[face_3_indices, :]

        # find surface Jacobian
        sJ = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= sJ
        ny /= sJ
        nz /= sJ
        sJ *= self.jacobians[face_node_indices, :]

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.surface_jacobians = sJ


    def _compute_face_node_mappings(self):
        # get constants
        Np = self.reference_element.nodes_per_cell
        Nfp = self.reference_element.nodes_per_face
        num_faces = self.reference_element.num_faces
        tolerance = self.reference_element.NODE_TOLERANCE
        CtoC = self.cell_to_cells
        CtoF = self.cell_to_faces
        K = self.num_cells
        
        # create global node ids
        node_ids = np.arange(K * Np).reshape(Np, K, order='F')

        # create interior exterior map matrices
        interior_face_node_indices = np.zeros((Nfp, num_faces, K), dtype=int)
        exterior_face_node_indices = np.zeros((Nfp, num_faces, K), dtype=int)
        
        # reshape face_mask
        face_node_indices = self.reference_element.face_node_indices.reshape(4, -1).T
        
        # Assign interior face node indices based on local face ordering
        for cell in range(K):
            for face in range(num_faces):
                interior_face_node_indices[:, face, cell] = node_ids[face_node_indices[:, face], cell]

        # Loop over each cell and its faces to establish exterior face node mapping
        for cell in range(K):
            for face in range(num_faces):
                # get neighbor cell and corresponding face
                adjacent_cell = CtoC[cell, face]
                adjacent_face = CtoF[cell, face]

                # Get interior face node indices for current and adjacent cell
                interior_face_node_ids = interior_face_node_indices[:, face, cell]
                exterior_face_node_ids = interior_face_node_indices[:, adjacent_face, adjacent_cell]

                # Retrieve the (x, y, z) coordinates of nodes on interior and exterior faces
                x_interior = np.ravel(self.x, order='F')[interior_face_node_ids][:, None]
                y_interior = np.ravel(self.y, order='F')[interior_face_node_ids][:, None]
                z_interior = np.ravel(self.z, order='F')[interior_face_node_ids][:, None]
                x_exterior = np.ravel(self.x, order='F')[exterior_face_node_ids][:, None]
                y_exterior = np.ravel(self.y, order='F')[exterior_face_node_ids][:, None]
                z_exterior = np.ravel(self.z, order='F')[exterior_face_node_ids][:, None]
                
                # Compute pairwise squared distance matrix between interior and exterior nodes
                node_distance = (x_interior - x_exterior.T)**2 + (y_interior - y_exterior.T)**2 + (z_interior - z_exterior.T)**2

                # Identify matching nodes based on small distance (within tolerance)
                interior_indices, exterior_indices = np.where(np.abs(node_distance) < tolerance)

                # Map interior face nodes to corresponding exterior face nodes
                exterior_face_node_indices[interior_indices, face, cell] = interior_face_node_indices[exterior_indices, adjacent_face, adjacent_cell]
                
        # Flatten the mappings for easy indexing
        self.exterior_face_node_indices = exterior_face_node_indices.reshape(-1, order='F')
        self.interior_face_node_indices = interior_face_node_indices.reshape(-1, order='F')


    def _find_boundary_nodes(self):
        # Identify boundary nodes (nodes with no adjacent exterior match)
        self.boundary_face_node_indices = np.where(self.exterior_face_node_indices == self.interior_face_node_indices)[0]
        self.boundary_node_indices= self.interior_face_node_indices[self.boundary_face_node_indices]

    def _compute_surface_to_volume_jacobian(self):
        sJ = self.surface_jacobians 
        face_node_indices = self.reference_element.face_node_indices
        J = self.jacobians
        self.surface_to_volume_jacobian = sJ / J[face_node_indices, :]

    def get_edges(self):
        # get edges
        edge_vertices = gmsh.model.mesh.getElementEdgeNodes(4)
        return edge_vertices.reshape(int(len(edge_vertices)/2), 2).astype(int) - 1

    def log_info(self):
        logger = getLogger("simlog")
        logger.info(f"Number of cells: {self.num_cells}")
        logger.info(f"Number of vertices: {self.num_vertices}")
        logger.info(f"Using {self.n} order Lagrange element")
        logger.info(f"Nodes per cell: {self.reference_element.nodes_per_cell}")
        logger.info(f"Number of nodes: {self.reference_element.nodes_per_cell * self.num_cells}")
