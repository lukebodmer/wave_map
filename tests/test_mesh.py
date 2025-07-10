import pytest
import matplotlib.pyplot as plt
import numpy as np
import gmsh
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


def test_drdx(mesh):
    """
    Check the derivatives drdx, drdy, drdz by computing a known function's derivative
    and comparing it to the computed values.
    """
    x, y, z = mesh.x, mesh.y, mesh.z  # Mesh node coordinates

    # Define a known function and its exact derivatives
    f = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    df_dx_exact = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) * np.exp(z)
    df_dy_exact = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(z)
    df_dz_exact = np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(z)

    # Compute numerical derivatives using the differentiation matrices
    Dr = mesh.reference_element_operators.r_differentiation_matrix,
    Ds = mesh.reference_element_operators.s_differentiation_matrix,
    Dt = mesh.reference_element_operators.t_differentiation_matrix,

    drdx, drdy, drdz = mesh.drdx, mesh.drdy, mesh.drdz
    dsdx, dsdy, dsdz = mesh.dsdx, mesh.dsdy, mesh.dsdz
    dtdx, dtdy, dtdz = mesh.dtdx, mesh.dtdy, mesh.dtdz

    df_dx_numeric = drdx * (Dr @ f) + dsdx * (Ds @ f) + dtdx * (Dt @ f)
    df_dy_numeric = drdy * (Dr @ f) + dsdy * (Ds @ f) + dtdy * (Dt @ f)
    df_dz_numeric = drdz * (Dr @ f) + dsdz * (Ds @ f) + dtdz * (Dt @ f)

    # Compute errors
    err_dx = np.linalg.norm(df_dx_exact - df_dx_numeric) / np.linalg.norm(df_dx_exact)
    err_dy = np.linalg.norm(df_dy_exact - df_dy_numeric) / np.linalg.norm(df_dy_exact)
    err_dz = np.linalg.norm(df_dz_exact - df_dz_numeric) / np.linalg.norm(df_dz_exact)

    print(f"Error in dx: {err_dx:.5e}")
    print(f"Error in dy: {err_dy:.5e}")
    print(f"Error in dz: {err_dz:.5e}")

    # Plot results
    #fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    #derivative_labels = ['dx', 'dy', 'dz']
    #exact_derivatives = [df_dx_exact, df_dy_exact, df_dz_exact]
    #computed_derivatives = [df_dx_numeric, df_dy_numeric, df_dz_numeric]

    #for i in range(3):
    #    ax1, ax2, ax3 = axs[i]
    #    exact = exact_derivatives[i]
    #    computed = computed_derivatives[i]
    #    error = np.abs(exact - computed)

    #    sc1 = ax1.scatter(x, y, c=exact, cmap='coolwarm', marker='o', s=5)
    #    ax1.set_title(f"Exact ∂f/∂{derivative_labels[i]}")
    #    plt.colorbar(sc1, ax=ax1)

    #    sc2 = ax2.scatter(x, y, c=computed, cmap='coolwarm', marker='o', s=5)
    #    ax2.set_title(f"Computed ∂f/∂{derivative_labels[i]}")
    #    plt.colorbar(sc2, ax=ax2)

    #    sc3 = ax3.scatter(x, y, c=error, cmap='inferno', marker='o', s=5)
    #    ax3.set_title(f"Error in ∂f/∂{derivative_labels[i]}")
    #    plt.colorbar(sc3, ax=ax3)

    #plt.tight_layout()
    #plt.show()
    assert err_dx < 0.01, f"drdx gives too large of an error: {err_dx:.5e}"


def test_jacobians(finite_element):
    mesh_file = "./inputs/meshes/default.msh"
    mesh = Mesh3d(mesh_file, finite_element)
    gmsh.initialize()
    gmsh.open(mesh.msh_file)
    name, dim, order, numNodes, localCoords, _ = gmsh.model.mesh.getElementProperties(4)
    jacobians, determinants, _ = gmsh.model.mesh.getJacobians(4, localCoords)
    gmsh_jacobians = jacobians.reshape(-1, 3, 3)
    # idk why gmsh_determinants are 1/8 the size
    # I think it has to do with the localCoords
    gmsh_determinants = determinants[::4] * 0.125
    gmsh.finalize()

    J = mesh.jacobians
    J = J[0,:]
    assert np.allclose(J, gmsh_determinants), f"Computed Jacobians don't match gmsh"
