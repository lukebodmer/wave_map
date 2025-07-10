import pytest
import matplotlib.pyplot as plt
import numpy as np
from wave_simulator.mesh import Mesh3d
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators

@pytest.fixture(scope="module")
def finite_element():
    return LagrangeElement(d=3, n=3)

@pytest.fixture(scope="module")
def reference_element_operators(finite_element):
    return ReferenceElementOperators(finite_element)

@pytest.fixture(scope="module")
def mesh(finite_element):
    mesh_file = "./inputs/meshes/default.msh"
    return Mesh3d(mesh_file, finite_element)

def test_vandermonde_3d_well_conditioned(reference_element_operators):
    """Check that the Vandermonde 3D matrix is well-conditioned."""
    V = reference_element_operators.vandermonde_3d
    cond_number = np.linalg.cond(V)
    
    assert cond_number < 1e6, f"Vandermonde matrix is poorly conditioned: {cond_number}"


def test_r_differentiation_matrix(mesh):
    # Assume we have access to the reference element differentiation matrix
    Dr = mesh.reference_element_operators.r_differentiation_matrix
    
    # Get reference element nodal coordinates
    r = mesh.reference_element.r  # 1D array of nodes in r-direction

    
    # Define the test function and its exact derivative
    f = np.sin(np.pi * r)  # Function: f(r) = sin(pi * r)
    df_exact = np.pi * np.cos(np.pi * r)  # Analytical derivative: df/dr = pi * cos(pi * r)
    
    # Compute the numerical derivative using the differentiation matrix
    df_num = Dr @ f  # Matrix-vector multiplication
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(r, df_exact, label="Exact Derivative", linestyle="none", marker="x", linewidth=2, color="black")
    plt.plot(r, df_num, label="Numerical Derivative", marker="o", linestyle="none", markersize=5, color="red")
    
    plt.xlabel("Reference Element r-coordinates")
    plt.ylabel("Derivative df/dr")
    plt.legend()
    plt.title("Comparison of Numerical and Exact Differentiation using $D_r$")
    plt.grid()
    plt.show()

    error_x = np.linalg.norm(df_num - df_exact) / np.linalg.norm(df_exact)
    print(f"Relative errors: dfdx: {error_x}")
    assert error_x < 1e6, f"R differentiation matrix's error too large: {error_x}"


def test_s_differentiation_matrix(mesh):
    # Assume we have access to the reference element differentiation matrix
    Ds = mesh.reference_element_operators.s_differentiation_matrix
    
    # Get reference element nodal coordinates
    s = mesh.reference_element.s  # 1D array of nodes in r-direction

    
    # Define the test function and its exact derivative
    f = np.sin(np.pi * s)  # Function: f(r) = sin(pi * r)
    df_exact = np.pi * np.cos(np.pi * s)  # Analytical derivative: df/dr = pi * cos(pi * r)
    
    # Compute the numerical derivative using the differentiation matrix
    df_num = Ds @ f  # Matrix-vector multiplication
    
    error_y = np.linalg.norm(df_num - df_exact) / np.linalg.norm(df_exact)
    print(f"Relative errors: dfdy: {error_y}")
    assert error_y < 1e6, f"S differentiation matrix's error too large: {error_y}"


def test_t_differentiation_matrix(mesh):
    # Assume we have access to the reference element differentiation matrix
    Dt = mesh.reference_element_operators.t_differentiation_matrix
    
    # Get reference element nodal coordinates
    t = mesh.reference_element.t  # 1D array of nodes in r-direction
    
    # Define the test function and its exact derivative
    f = np.sin(np.pi * t)  # Function: f(r) = sin(pi * r)
    df_exact = np.pi * np.cos(np.pi * t)  # Analytical derivative: df/dr = pi * cos(pi * r)
    
    # Compute the numerical derivative using the differentiation matrix
    df_num = Dt @ f  # Matrix-vector multiplication
    
    error_z = np.linalg.norm(df_num - df_exact) / np.linalg.norm(df_exact)
    print(f"Relative errors: dfdy: {error_z}")
    assert error_z < 1e6, f"S differentiation matrix's error too large: {error_z}"

import numpy as np

import numpy as np

def test_lift_matrix(mesh):
    """
    Tests whether the lift matrix correctly computes the surface integral of a function over a triangle.
    
    Parameters:
        mesh: Mesh object containing the reference element operators.
        
    Returns:
        None. Prints the computed and exact integrals along with the error.
    """
    # Get necessary mesh components
    lift = mesh.reference_element_operators.lift_matrix
    face_scale = mesh.surface_to_volume_jacobian  # Surface scaling factors
    nodes_per_face = mesh.reference_element.nodes_per_face
    num_faces = mesh.reference_element.num_faces
    nodes_per_cell = mesh.reference_element.nodes_per_cell
    
    # Define a test function f(x, y, z) over the reference element
    x, y, z = mesh.x, mesh.y, mesh.z
    f = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    
    # Compute the exact integral over the reference faces
    exact_integrals = np.zeros((num_faces,))

    face_node_indices = mesh.reference_element.face_node_indices.reshape(4, -1).T
    for face in range(num_faces):
        face_indices = face_node_indices[:, face]
        face_x, face_y, face_z = x[face_indices], y[face_indices], z[face_indices]
        face_f = np.sin(np.pi * face_x) * np.cos(np.pi * face_y) * np.exp(face_z)
        face_jacobian = face_scale[face_indices]  # Surface Jacobian
        exact_integrals[face] = np.sum(face_f * face_jacobian) / nodes_per_face  # Approximate integral using quadrature

    # Compute the numerical surface integral using the lift matrix
    f_surface = np.zeros((nodes_per_face * num_faces,))
    for face in range(num_faces):
        face_indices = face_node_indices[:, face]
        breakpoint()
        f_surface[face * nodes_per_face:(face + 1) * nodes_per_face] = f[face_indices]
    
    computed_integral = lift @ (face_scale * f_surface / 2)

    # Compare results
    exact_integral_total = np.sum(exact_integrals)
    computed_integral_total = np.sum(computed_integral)

    error = np.abs(exact_integral_total - computed_integral_total) / np.abs(exact_integral_total)

    print(f"Exact surface integral: {exact_integral_total:.6e}")
    print(f"Computed surface integral: {computed_integral_total:.6e}")
    print(f"Relative error: {error:.6e}")

    return computed_integral, exact_integrals
   

#def test_r_differentiation_matrix(mesh):
#    # Get mesh node coordinates
#    x, y, z = mesh.x, mesh.y, mesh.z  
#
#    # Define test function
#    f = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
#
#    # Compute numerical derivatives using differentiation matrices
#    Dr = mesh.reference_element_operators.r_differentiation_matrix
#    Ds = mesh.reference_element_operators.r_differentiation_matrix
#    Dt = mesh.reference_element_operators.r_differentiation_matrix
#
#    dfdx_num = mesh.drdx * (Dr @ f) + mesh.dsdx * (Ds @ f) + mesh.dtdx * (Dt @ f)
#    dfdy_num = mesh.drdy * (Dr @ f) + mesh.dsdy * (Ds @ f) + mesh.dtdy * (Dt @ f)
#    dfdz_num = mesh.drdz * (Dr @ f) + mesh.dsdz * (Ds @ f) + mesh.dtdz * (Dt @ f)
#
#    # Compute analytical derivatives
#    dfdx_exact = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
#    dfdy_exact = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
#    dfdz_exact = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
#
#    # Compute errors
#    error_x = np.linalg.norm(dfdx_num - dfdx_exact) / np.linalg.norm(dfdx_exact)
#    error_y = np.linalg.norm(dfdy_num - dfdy_exact) / np.linalg.norm(dfdy_exact)
#    error_z = np.linalg.norm(dfdz_num - dfdz_exact) / np.linalg.norm(dfdz_exact)
#
#    print(f"Relative errors: dfdx: {error_x}, dfdy: {error_y}, dfdz: {error_z}")
#    assert error_x < 1e6, f"R differentiation matrix's error too large: {error_x}"
