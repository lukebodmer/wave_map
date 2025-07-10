import sympy as sp

# Define symbolic variables
rho, c = sp.symbols('rho, c')
rho_p, c_p = sp.symbols('rho_p, c_p')
rho_m, c_m = sp.symbols('rho_m, c_m')
nx, ny, nz = sp.symbols('nx, ny, nz')
tx, ty, tz = sp.symbols('tx, ty, tz')
sx, sy, sz = sp.symbols('sx, sy, sz')
p_m, u_m, v_m, w_m = sp.symbols('p_m, u_m, v_m, w_m')
p_p, u_p, v_p, w_p = sp.symbols('p_p, u_p, v_p, w_p')
udotn_p, udotn_m = sp.symbols('udotn_p, udotn_m')
pressure_jump, normal_jump = sp.symbols('pressure_jump, normal_jump')

# Define matrices
T2 = sp.Matrix([[1, 0, 0],
                [0, nx, -nz],
                [0, nz, nx]])

T2_inv = sp.Matrix([[1, 0, 0],
                    [0, nx, nz],
                    [0, -nz, nx]])

T = sp.Matrix([[1, 0, 0, 0],
               [0, nx, sx, tx],
               [0, ny, sy, ty],
               [0, nz, sz, tz]])

correction = sp.Matrix([rho_m * c_m**2 * (p_p - p_m - rho_p * c_p * (udotn_p - udotn_m))/(rho_m * c_m + rho_p * c_p),
                        -c_m * (p_p - p_m - rho_p * c_p * (udotn_p - udotn_m))/(rho_m * c_m + rho_p * c_p),
                        0,
                        0])
FW_m = sp.Matrix([
    [rho_m * c_m**2 * u_m, rho_m * c_m**2 * v_m, rho_m * c_m**2 * w_m],
    [p_m / rho_m, 0, 0],
    [0, p_m / rho_m, 0],
    [0, 0, p_m / rho_m]
])

normal = sp.Matrix([nx, ny, nz])

A1 = sp.Matrix([[0, rho_m * c_m**2],[1/rho_m, 0]])
An2_m = sp.Matrix([[0, rho_m * c_m**2, 0], [1/rho_m, 0, 0], [0,0,0]])
Bn2_p = sp.Matrix([[0, 0, rho_m * c_m**2], [0, 0, 0], [1/rho_m,0,0]])
A_hat = An2_m * nx + Bn2_p * ny

An_m = sp.Matrix([[0, rho_m * c_m**2, 0, 0], [1/rho_m, 0, 0, 0], [0,0,0,0], [0,0,0,0]])
Bn_m = sp.Matrix([[0, 0, rho_m * c_m**2, 0], [0, 0, 0, 0], [1/rho_m,0,0,0], [0,0,0,0]])
Cn_m = sp.Matrix([[0, 0, 0, rho_m * c_m**2], [0, 0, 0, 0], [0,0,0,0], [1/rho_m,0,0,0]])

An_p = sp.Matrix([[0, rho_p * c_p**2, 0, 0], [1/rho_p, 0, 0, 0], [0,0,0,0], [0,0,0,0]])
Bn_p = sp.Matrix([[0, 0, rho_p * c_p**2, 0], [0, 0, 0, 0], [1/rho_p,0,0,0], [0,0,0,0]])
Cn_p = sp.Matrix([[0, 0, 0, rho_p * c_p**2], [0, 0, 0, 0], [0,0,0,0], [1/rho_p,0,0,0]])

An = sp.Matrix([[0, rho * c**2, 0, 0], [1/rho, 0, 0, 0], [0,0,0,0], [0,0,0,0]])
Bn = sp.Matrix([[0, 0, rho * c**2, 0], [0, 0, 0, 0], [1/rho,0,0,0], [0,0,0,0]])
Cn = sp.Matrix([[0, 0, 0, rho * c**2], [0, 0, 0, 0], [0,0,0,0], [1/rho,0,0,0]])

A_hat = An * nx + Bn * ny + Cn * nz
A_hat_m = An_m * nx + Bn_m * ny + Cn_m * nz
A_hat_p = An_p* nx + Bn_p * ny + Cn_p * nz
A_hat_avg = (An_p * nx + Bn_p * ny + Cn_p * nz +  An_m * nx + Bn_m * ny + Cn_m * nz)/2

R = sp.Matrix([[-c_m * rho_m, 0, 0, c_p * rho_p],
               [nx, -ny, -nz, nx],
               [ny, nx, 0, ny],
               [nz, 0, nx, nz]])

R_inv = sp.Matrix([[-1/(c_m*rho_m + c_p*rho_p), nx*c_p*rho_p/(c_m * rho_m + c_p*rho_p), ny*c_p*rho_p/(c_m * rho_m + c_p*rho_p), nz*c_p*rho_p/(c_m * rho_m + c_p*rho_p)],
                   [0, -ny, (nx**2 + nz**2)/nx, -ny*nz/nx],
                   [0, -nz, -ny*nz/nx, (nx**2+ny**2)/nx],
                   [1/(c_m*rho_m + c_p*rho_p), nx*c_m*rho_m/(c_m * rho_m + c_p*rho_p), ny*c_m*rho_m/(c_m * rho_m + c_p*rho_p), nz*c_m*rho_m/(c_m * rho_m + c_p*rho_p)]])

q_m = sp.Matrix([p_m, u_m, v_m, w_m])
q_p = sp.Matrix([p_p, u_p, v_p, w_p])

P, lambda_p = A_hat_p.diagonalize()
M, lambda_m = A_hat_m.diagonalize()

alpha = R.inv() * (q_p - q_m)

#flux = -c_m * alpha[0,:] * R[:,0].T + c_p * alpha[3,:] * R[:,3].T

flux1 = (A_hat_m * q_m).T + c_m * alpha[0,:] * R[:,0].T 
flux2 = (A_hat_p * q_p).T + c_p * alpha[3,:] * R[:,3].T   
breakpoint()
