import numpy as np
from scipy.special import gamma
from scipy.linalg import eig


class LagrangeElement:
    """Lagrange finite element defined on a triangle and
      tetrahedron"""
    def __init__(self, d, n):
        self.d = d  # dimension
        self.n = n  # polynomial order
        self.nodes_per_cell: int = (n + 1) * (n + 2) * (n + 3) // 6
        self.nodes_per_face: int = (n + 1) * (n + 2) // 2
        #self.num_faces = 0  # no. faces per element
        #self.nodes = np.array([])
        #self.r = np.array([])
        #self.s = np.array([])
        #self.t = np.array([])
        #self.vertices = np.array([])
        #self.face_node_indices = np.array([]) 
        self.NODE_TOLERANCE = 1e-7  # tolerance to find face nodes

        self._get_nodes_and_vertices()
        self._find_face_nodes()

    def _get_nodes_and_vertices(self):
        """ get tabulated data for nodes, and mesh data for vertices"""
        n = self.n
        # tetrahedron
        self.num_faces = 4
        #self.nodes = self._tetrahedron_nodes[str(n)]
        self._compute_warp_and_blend_nodes()
        self.vertices = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [-1,  1, -1],
            [-1, -1,  1]
        ])


    def _compute_warp_and_blend_nodes(self):
        """
        Compute Warp & Blend nodes
        Input: p = polynomial order of interpolant
        Output: x, y, z = vectors of node coordinates in an equilateral tetrahedron
        """
        # Create equidistributed nodes
        equidistant_r, equidistant_s, equidistant_t = self._equidistributed_nodes_3d(self.n)
        
        # get barycentric coordinates (lambdas) in terms of r,s,t
        lambda_1, lambda_2, lambda_3, lambda_4 = self._barycentric_coordinates(equidistant_r, equidistant_s, equidistant_t)
        
        # get vertices of equilateral tetrahedron
        v1, v2, v3, v4 = self._equilateral_tetrahedron_vertices()
        
        # find tangent vectors at each face
        t1, t2 = self._face_tangents(v1, v2, v3, v4)
        
        # Form undeformed coordinates
        unwarped_xyz = lambda_3 @ v1 + lambda_4 @ v2 + lambda_2 @ v3 + lambda_1 @ v4
        
        # Warp and blend for each face (accumulated in shiftXYZ)
        nodes = self._warp_and_blend(unwarped_xyz, t1, t2, lambda_1, lambda_2, lambda_3, lambda_4)
        
        x = nodes[:, 0]
        y = nodes[:, 1]
        z = nodes[:, 2]

        r, s, t = self._map_equilateral_to_reference_tetrahedron(x, y, z)
        self.r = r
        self.s = s
        self.t = t
        self.nodes = np.column_stack([r,s,t])

    def _equidistributed_nodes_3d(self, polynomial_order):
        """Compute the equidistributed node coordinates on the reference tetrahedron"""
    
        # Total number of nodes
        # Create equidistributed nodes on the equilateral triangle
        x = np.zeros(self.nodes_per_cell)
        y = np.zeros(self.nodes_per_cell)
        z = np.zeros(self.nodes_per_cell)
        polynomial_order = self.n
    
        node_index = 0
        for n in range(1, polynomial_order+2):
            for m in range(1, polynomial_order+3-n):
                for q in range(1, polynomial_order+4-n-m):
                    x[node_index] = -1 + (q-1)*2/polynomial_order
                    y[node_index] = -1 + (m-1)*2/polynomial_order
                    z[node_index] = -1 + (n-1)*2/polynomial_order
                    node_index += 1
    
        return x, y, z


    def _barycentric_coordinates(self, r, s, t):
        """ return the barycentric coordinates for a given r, s, t"""
        L1 = (1+t)/2
        L2 = (1+s)/2
        L3 = -(1+r+s+t)/2
        L4 = (1+r)/2
        return L1.reshape(-1, 1), L2.reshape(-1, 1), L3.reshape(-1, 1), L4.reshape(-1, 1)
    

    def _equilateral_tetrahedron_vertices(self):
        """ Creates the 3D coordinates for the vertices of an
        equilateral tetrahedron """
        v1 = np.array([-1, -1/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
        v2 = np.array([1, -1/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
        v3 = np.array([0, 2/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
        v4 = np.array([0, 0, 3/np.sqrt(6)]).reshape(1, 3)
        return v1, v2, v3, v4


    def _face_tangents(self, v1, v2, v3, v4):
        """ find the two orthogonal tangent vectors to each face """
        
        # create t1 and t2 vectors
        t1 = np.zeros((4, 3))
        t2 = np.zeros((4, 3))

        # compute tangent vectors
        t1[0, :] = v2 - v1
        t1[1, :] = v2 - v1
        t1[2, :] = v3 - v2
        t1[3, :] = v3 - v1
        t2[0, :] = v3 - 0.5*(v1 + v2)
        t2[1, :] = v4 - 0.5*(v1 + v2)
        t2[2, :] = v4 - 0.5*(v2 + v3)
        t2[3, :] = v4 - 0.5*(v1 + v3)
        
        # Normalize tangents
        for n in range(4):
            t1[n, :] = t1[n, :] / np.linalg.norm(t1[n, :])
            t2[n, :] = t2[n, :] / np.linalg.norm(t2[n, :])

        return t1, t2


    def _warp_and_blend(self, xyz, t1, t2, L1, L2, L3, L4):
        """ warp and blend the equidistant nodes on an equilateral tetrahedron
        to the generalization of Legendre-Gauss_Lobatto points"""
        N = self.n
    
        # Choose optimized blending parameter
        alpha = self._get_alpha_value(N)
    
        # declare shift variable that will add to xyz
        shift = np.zeros_like(xyz)
    
        # tolerance to avoid dividing by zero
        tol = 1e-10
    
        # calculate warp amount for each face
        for face in range(4):
            La, Lb, Lc, Ld = self._corresponding_lambdas(face, L1, L2, L3, L4)
    
            # Compute warp tangential to the face
            warp1, warp2 = self._warp_shift_face_3d(N, alpha, Lb, Lc, Ld)
    
            # Compute volume blending
            blend = Lb * Lc * Ld
    
            # Modify linear blend
            denominator = (Lb + 0.5 * La) * (Lc + 0.5 * La) * (Ld + 0.5 * La)
            ids = np.where(denominator > tol)[0]
            blend[ids] = (1 + (alpha * La[ids]) ** 2) * blend[ids] / denominator[ids]
    
            # Compute warp & blend
            shift += (blend * warp1) @ np.reshape(t1[face, :], (1,3)) + (blend * warp2) @ np.reshape(t2[face, :], (1, 3))
    
            # Fix face warp
            ids = np.where((La < tol) & (np.array(Lb > tol, dtype=int) + np.array(Lc > tol, dtype=int) + np.array(Ld > tol, dtype=int) < 3))[0]
    
            shift[ids, :] = warp1[ids, :] * t1[face, :] + warp2[ids, :] * t2[face, :]
    
        # add warp and bend shift to XYZ
        xyz += shift
        return xyz

    
    def _get_alpha_value(self, N):
        """ function that stores optimal blending parameter as calculated in
        Hesthaven, Warburton - Nodal DG methods """
    
        alpha_store = [0, 0, 0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
                       1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655]
    
        # If N is greater than 15, alpha = 1 is a good enough approximation.
        if N <= 15:
            alpha = alpha_store[N-1]
        else:
            alpha = 1.0
        return alpha

    def _corresponding_lambdas(self, face_number, L1, L2, L3, L4):
        """ select the correct barycentric coordinates for each face"""
        if face_number == 0:
            return L1, L2, L3, L4
        elif face_number == 1:
            return L2, L1, L3, L4
        elif face_number == 2:
            return L3, L1, L4, L2
        elif face_number == 3:
            return L4, L1, L3, L2

    def _warp_shift_face_3d(self, p, pval, L2, L3, L4):
        """ compute warp factor used in creating 3D Warp & Blend nodes """
    
        dtan1, dtan2 = self._eval_shift(p, pval, L2, L3, L4)
        warpx = dtan1
        warpy = dtan2
    
        return warpx, warpy


    def _eval_shift(self, p, pval, L1, L2, L3):
        """Compute two-dimensional Warp & Blend transform"""
    
        # 1) Compute Gauss-Lobatto-Legendre node distribution
        gaussX = -self._gauss_lobatto_quadrature_points(0, 0, p)
    
        # 2) Compute blending function at each node for each edge
        blend1 = L2 * L3
        blend2 = L1 * L3
        blend3 = L1 * L2
    
        # 3) Amount of warp for each node, for each edge
        warpfactor1 = 4 * self._compute_edge_warp(p, gaussX, L3 - L2)
        warpfactor2 = 4 * self._compute_edge_warp(p, gaussX, L1 - L3)
        warpfactor3 = 4 * self._compute_edge_warp(p, gaussX, L2 - L1)
    
        # 4) Combine blend & warp
        warp1 = blend1 * warpfactor1 * (1 + (pval * L1) ** 2)
        warp2 = blend2 * warpfactor2 * (1 + (pval * L2) ** 2)
        warp3 = blend3 * warpfactor3 * (1 + (pval * L3) ** 2)
    
        # 5) Evaluate shift in equilateral triangle
        dx = 1 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
        dy = 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3
    
        return dx, dy


    def _compute_edge_warp(self, p, xnodes, xout):
        """Compute one-dimensional edge warping function"""
    
        warp = np.zeros_like(xout)
    
        xeq = np.zeros(p+1)
        for i in range(p+1):
            xeq[i] = -1 + 2*(p-i)/p
    
        for i in range(1, p + 2):
            d = (xnodes[i - 1] - xeq[i - 1])
    
            for j in range(2, p + 1):
                if i != j:
                    d *= (xout - xeq[j - 1]) / (xeq[i - 1] - xeq[j - 1])
    
            if i != 1:
                d = -d / (xeq[i - 1] - xeq[0])
    
            if i != (p + 1):
                d = d / (xeq[i - 1] - xeq[p])
    
            warp += d
    
        return warp


    def eval_2d_basis_function(self, r, s, i, j):
        """ Evaluate 2D orthonormal polynomial on simplex at (a,b) of order (i,j) """
        a, b = self._rs_to_ab(r,s)

        h1 = self.eval_jacobi_polynomial(a, 0, 0, i)
        h2 = self.eval_jacobi_polynomial(b, 2 * i + 1, 0, j)

        P = np.sqrt(2.0) * h1 * h2 * (1 - b) ** i
        return P


    def eval_3d_basis_function(self, r, s, t, i, j, k):
        """ evaulated orthonormal basis function polynomials """
        a, b, c = self._rst_to_abc(r,s,t)

        h1 = self.eval_jacobi_polynomial(a, 0, 0, i)
        h2 = self.eval_jacobi_polynomial(b, 2*i+1, 0, j)
        h3 = self.eval_jacobi_polynomial(c, 2*(i+j)+2, 0, k)
        
        P = 2 * np.sqrt(2) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))
        return P



    def eval_3d_basis_function_gradient(self, r, s, t, id, jd, kd):
        """ Return the derivatives of the modal basis (id,jd,kd) on the 3D simplex at (a,b,c)"""
        
        a, b, c = self._rst_to_abc(r,s,t)

        fa = self.eval_jacobi_polynomial(a, 0, 0, id)
        dfa = self.evaluate_jacobi_polynomial_gradient(a, 0, 0, id)
        gb = self.eval_jacobi_polynomial(b, 2*id+1, 0, jd)
        dgb = self.evaluate_jacobi_polynomial_gradient(b, 2*id+1, 0, jd)
        hc = self.eval_jacobi_polynomial(c, 2*(id+jd)+2, 0, kd)
        dhc = self.evaluate_jacobi_polynomial_gradient(c, 2*(id+jd)+2, 0, kd)

        # calculate r derivative Vr
        Vr = dfa * (gb * hc)
        if id > 0:
            Vr *= (0.5 * (1 - b)) ** (id - 1)
        if id + jd > 0:
            Vr *= (0.5 * (1 - c)) ** (id + jd - 1)

        # calculate s derivative Vs
        Vs = 0.5 * (1 + a) * Vr
        tmp = dgb * ((0.5 * (1 - b)) ** id)
        if id > 0:
            tmp += (-0.5 * id) * (gb * ((0.5 * (1 - b)) ** (id - 1)))
        if id + jd > 0:
            tmp *= (0.5 * (1 - c)) ** (id + jd - 1)
        tmp = fa * (tmp * hc)
        Vs += tmp

        # calculate t derivative Vt
        Vt = 0.5 * (1 + a) * Vr + 0.5 * (1 + b) * tmp
        tmp = dhc * ((0.5 * (1 - c)) ** (id + jd))
        if id + jd > 0:
            tmp -= 0.5 * (id + jd) * (hc * ((0.5 * (1 - c)) ** (id + jd - 1)))
        tmp = fa * (gb * tmp)
        tmp *= (0.5 * (1 - b)) ** id
        Vt += tmp

        # normalize
        Vr *= 2 ** (2*id+jd+1.5)
        Vs *= 2 ** (2*id+jd+1.5)
        Vt *= 2 ** (2*id+jd+1.5)

        return Vr, Vs, Vt

    def _rst_to_abc(self, r, s, t):
        """ transfer from (r,s,t) coordinates to (a,b,c) which are used to evaluate the
        jacobi polynomials in our orthonormal basis """
        Np = len(r)
        a = np.zeros(Np)
        b = np.zeros(Np)
        c = np.zeros(Np)
        
        for n in range(Np):
            if s[n] + t[n] != 0:
                a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1
            else:
                a[n] = -1

            if t[n] != 1:
                b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
            else:
                b[n] = -1

            c[n] = t[n]
        
        return a, b, c


    def _rs_to_ab(self, r, s):
        """ map from (r,s) coordinates on an element face to (a,b) which are used to evaluate the
            jacobi polynomials in our orthonormal basis """
        Np = len(r)
        a = np.zeros(Np)
        for n in range(Np):
            if s[n] != 1:
                a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
            else:
                a[n] = -1
        b = s
        return a, b

    def _map_equilateral_to_reference_tetrahedron(self, x, y, z):
        """ map x,y,z to the standard tetrahedron """
    
        # define vertices of standard tetrahedron
        v1 = np.array([-1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
        v2 = np.array([1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
        v3 = np.array([0, 2 / np.sqrt(3), -1 / np.sqrt(6)])
        v4 = np.array([0, 0 / np.sqrt(3), 3 / np.sqrt(6)])
    
        # subtract the average coordinates of the vertices v1, v2, v3, and v4
        # (with appropriate adjustments) from the given X, Y, and Z coordinates.
        # This step aligns the coordinates with the reference tetrahedron.
        rhs = (np.array([x, y, z]).T - np.array([0.5 * (v2 + v3 + v4 - v1)])).T
    
        # solve matrix equation for mapping on pg 410
        A = np.column_stack(
            [0.5 * (v2 - v1), 0.5 * (v3 - v1), 0.5 * (v4 - v1)])
    
        RST = np.linalg.solve(A, rhs)
    
        r = RST[0, :].T
        s = RST[1, :].T
        t = RST[2, :].T
    
        return r, s, t


    def eval_jacobi_polynomial(self, x, alpha, beta, N):
        # Turn points into row if needed.
        xp = x
        
        # PL will carry our values for the jacobi polynomial
        PL = np.zeros((N+1, len(x)))
        
        # initialize values P_0(x)
        gamma0 = 2 ** (alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * \
            gamma(beta + 1) / gamma(alpha + beta + 1)
        PL[0, :] = 1.0 / np.sqrt(gamma0)
        
        # return if N = 0
        if N == 0:
            P = PL[0, :]
            return P
        
        # initialize value P_1(x)
        gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
        PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
        
        # return if N = 1
        if N == 1:
            P = PL[N, :]
            return P
        
        # Repeat value in recurrence.
        a_old = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
        
        # Forward recurrence using the symmetry of the recurrence.
        for i in range(1, N):
            h1 = 2 * i + alpha + beta
            a_new = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) *
                                           (i + 1 + beta) / (h1 + 1) / (h1 + 3))
            b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
            PL[i + 1, :] = 1 / a_new * (-a_old * PL[i-1, :] + (xp - b_new) * PL[i, :])
            a_old = a_new
            
        #P = np.reshape(PL[N, :], (np.shape(PL[N, :])[0], 1)).T
        P = PL[N, :]
        return P
        
        
    def _jacobi_gauss_quadrature_points(self, alpha, beta, N):
        """Compute the N'th order Gauss quadrature points, x,
        and weights, w, associated with the Jacobi polynomial
        of type (alpha, beta) > -1 ( <> -0.5)."""
            
        x = np.zeros(N+1)
        w = np.zeros(N+1)
            
        if N == 0:
            x[0] = -(alpha-beta)/(alpha+beta+2)
            w[0] = 2
            return x, w
            
        # Form symmetric matrix from recurrence
        h1 = 2 * (np.arange(N+1)) + alpha + beta
        J = np.diag(-1/2 * (alpha**2 - beta**2) / (h1 + 2) / h1) + \
            np.diag(2 / (h1[0:N] + 2) * np.sqrt((np.arange(1, N+1)) * ((np.arange(1, N+1)) + alpha + beta) * \
                                                ((np.arange(1, N+1)) + alpha) * ((np.arange(1, N+1)) + beta) / \
                                                (h1[0:N] + 1) / (h1[0:N] + 3)), 1)
            
        if alpha + beta < 10 * np.finfo(float).eps:
            J[0, 0] = 0.0
            
        J = J + J.T
                
        # Compute quadrature by eigenvalue solve
        D, V = eig(J)
        sorted_indices = np.argsort(D)
        D = D[sorted_indices]
        V = V[:, sorted_indices]
        x = np.real(D)
            
        w = np.real((V[0, :]**2) * 2**(alpha + beta + 1) / (alpha + beta + 1) * \
                    gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1))
            
        return x, w
            
            
    def _gauss_lobatto_quadrature_points(self, alpha, beta, N):
        """ Compute the N'th order Gauss Lobatto quadrature points, x,
            associated with the Jacobi polynomial of type (alpha, beta) > -1 ( <> -0.5)."""
                
        x = np.zeros(N+1)
        if N == 1:
            x[0] = -1.0
            x[1] = 1.0
            return x
                
        xint, w = self._jacobi_gauss_quadrature_points(alpha+1, beta+1, N-2)
        x[0] = -1
        x[1:N] = xint
        x[N] = 1
                
        return x
            
            
    def evaluate_jacobi_polynomial_gradient(self, r, alpha, beta, N):
        dP = np.zeros(len(r))
        if N == 0:
            dP[:] = 0.0
        else:
            dP = np.sqrt(N * (N + alpha + beta + 1)) * self.eval_jacobi_polynomial(r, alpha + 1, beta + 1, N - 1)
        return dP       


    def _find_face_nodes(self):
        """ return node indexes for nodes on each face of the standard tetrahedron"""
        tolerance = self.NODE_TOLERANCE
        r = self.r
        s = self.s
        t = self.t

        face_0_indices = np.where(np.abs(1 + t) < tolerance)[0]
        face_1_indices = np.where(np.abs(1 + s) < tolerance)[0]
        face_2_indices = np.where(np.abs(1 + r + s + t) < tolerance)[0]
        face_3_indices = np.where(np.abs(1 + r) < tolerance)[0]

        face_node_indices = np.concatenate((face_0_indices,
                                            face_1_indices,
                                            face_2_indices,
                                            face_3_indices))

        self.face_node_indices = face_node_indices


