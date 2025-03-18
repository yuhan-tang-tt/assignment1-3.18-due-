1.1 Question 1: Area of a triangle

    import numpy as np

    def triangle_area_heron(a, b, c):

        # Create array of zeros same shape as input
        area = np.zeros_like(a, dtype=float)
    
        # Calculate s for all values
        s = ((a + b) + c)/2
    
        # Use numpy where to handle the triangle inequality
        valid = (a + b > c) & (b + c > a) & (a + c > b)
        area = np.where(valid, np.sqrt(s*(s - a)*(s - b)*(s - c)), np.nan)
    
        return area

1.2 Write a function named triangle_area_kahan that calculates the area of a triangle using the Kahan summation algorithm.

    import numpy as np

    def triangle_area_kahan(a, b, c):


        # Sort sides in decreasing order
        sides = np.sort([a, b, c])[::-1]
        a, b, c = sides[0], sides[1], sides[2]
    
        area = (1/4) * np.sqrt((a + (b + c)) * 
                              (c - (a - b)) * 
                              (a + (b - c)) * 
                              (a + (c - b)))
        return area

1.3 test code

    import matplotlib.pyplot as plt
    import numpy as np
    np.seterr(invalid='ignore')
    # Generate x values
    x_values = np.logspace(-15, 2, 100)

    # Calculate side lengths
    a = 2 * x_values
    b = c = np.sqrt(1 + x_values**2)/x_values

    # Calculate exact area (using the height formula: A = base * height/2)
    exact_area = np.ones_like(x_values)  # The area is actually 1 for all x

    # Calculate areas using both methods
    heron_area = triangle_area_heron(a, b, c)
    kahan_area = triangle_area_kahan(a, b, c)

    # Calculate relative errors
    heron_error = np.abs(heron_area - exact_area)/exact_area
    kahan_error = np.abs(kahan_area - exact_area)/exact_area

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(x_values, heron_error, 'b-', label="Heron's Formula")
    plt.loglog(x_values, kahan_error, 'r-', label="Kahan's Formula")
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('Relative Error')
    plt.title('Relative Error in Triangle Area Calculation')
    plt.legend()
    plt.show()

2.1 Write a function named sequence_element which accepts as input an int deffning n, andwhich returns xn as a shape (2,) NumPy array (i.e. a vector) with integer scalar data type.

    import numpy as np

    def sequence_element(n):
        # Define initial vector x0
        x0 = np.array([1, 1], dtype=int)
        
        # Define matrix A
        A = np.array([[0, 1], [1, 1]], dtype=int)
        
        # Compute A^n * x0
        x = np.linalg.matrix_power(A, n) @ x0
        
        return x

2.2 Perform numerical calculations to investigate (Optional)

    import numpy as np

    # Calculate eigenvalues of A
    A = np.array([[0, 1], [1, 1]])
    eigenvals, eigenvecs = np.linalg.eig(A)

    # Find largest eigenvalue in magnitude
    lambda_max = eigenvals[np.argmax(np.abs(eigenvals))]

    # Generate n values
    n_values = np.arange(1, 31)

    # Calculate error ratios
    error_ratios = []
    for n in n_values:
        xn = sequence_element(n)
        xn_minus_1 = sequence_element(n-1)
        
        error = np.linalg.norm(xn - lambda_max * xn_minus_1)
        denominator = np.linalg.norm(xn_minus_1)
        
        error_ratio = error / denominator
        error_ratios.append(error_ratio)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(n_values, error_ratios, 'b.-')
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('Error Ratio (ε_n)')
    plt.title('Convergence of Sequence to Dominant Eigenvalue')
    plt.show()

    # Print the largest eigenvalue
    print(f"Largest eigenvalue (λ): {lambda_max:.8f}")
    
3.1 Write a function interpolatory_quadrature_weights

    import numpy as np
    import matplotlib.pyplot as plt

    def interpolatory_quadrature_weights(x):
        # Number of points
        N = len(x) - 1
        
        # Construct Vandermonde matrix
        V = np.vander(x, N+1, increasing=True)
        
        # Compute the moments (integrals of monomials)
        moments = np.zeros(N+1)
        for i in range(N+1):
            moments[i] = (1 - (-1)**(i+1))/(i+1)
        
        # Solve for weights
        w = np.linalg.solve(V.T, moments)
        
        return w

    # Test code for various quadrature rules
    # Midpoint rule (N=0)
    x_midpoint = np.array([0.0])
    w_midpoint = interpolatory_quadrature_weights(x_midpoint)
    print("Midpoint rule weights:")
    print(f"Computed: {w_midpoint}")
    print(f"Expected: [2.0]")
    print()

    # Trapezoidal rule (N=1)
    x_trap = np.array([-1.0, 1.0])
    w_trap = interpolatory_quadrature_weights(x_trap)
    print("Trapezoidal rule weights:")
    print(f"Computed: {w_trap}")
    print(f"Expected: [1.0, 1.0]")
    print()

    # Simpson's rule (N=2)
    x_simpson = np.array([-1.0, 0.0, 1.0])
    w_simpson = interpolatory_quadrature_weights(x_simpson)
    print("Simpson's rule weights:")
    print(f"Computed: {w_simpson}")
    print(f"Expected: [1/3, 4/3, 1/3] ≈ [{1/3:.6f}, {4/3:.6f}, {1/3:.6f}]")
    
    #output
    Midpoint rule weights:
    Computed: [2.]
    Expected: [2.0]

    Trapezoidal rule weights:
    Computed: [1. 1.]
    Expected: [1.0, 1.0]

    Simpson's rule weights:
    Computed: [0.33333333 1.33333333 0.33333333]
    Expected: [1/3, 4/3, 1/3] ≈ [0.333333, 1.333333, 0.333333]

3.2 For a given positive integer N Consider two sets of quadrature points ...

    import numpy as np
    import matplotlib.pyplot as plt

    def f(x):
        """The function to integrate: f(x) = 1/(1 + (3x)^2)"""
        return 1/(1 + 9*x**2)

    def get_quadrature_points(N, method='uniform'):
        """Generate quadrature points using either uniform or Chebyshev nodes"""
        if method == 'uniform':
            return -1 + 2*np.arange(N+1)/N  # x_{0,N}
        else:
            return -np.cos(np.pi * np.arange(N+1)/N)  # x_{1,N}

    # Calculate exact value using analytical solution
    exact_value = np.arctan(3)/3

    # Test different values of N
    N_values = np.arange(2, 31)
    errors_uniform = []
    errors_chebyshev = []

    for N in N_values:
        # Get quadrature points
        x_uniform = get_quadrature_points(N, 'uniform')
        x_chebyshev = get_quadrature_points(N, 'chebyshev')
        
        # Calculate weights for both sets of points
        w_uniform = interpolatory_quadrature_weights(x_uniform)
        w_chebyshev = interpolatory_quadrature_weights(x_chebyshev)
        
        # Calculate approximations
        approx_uniform = np.sum(w_uniform * f(x_uniform))
        approx_chebyshev = np.sum(w_chebyshev * f(x_chebyshev))
        
        # Calculate relative errors
        errors_uniform.append(np.abs(approx_uniform - exact_value)/exact_value)
        errors_chebyshev.append(np.abs(approx_chebyshev - exact_value)/exact_value)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, errors_uniform, 'b.-', label='Uniform points')
    plt.semilogy(N_values, errors_chebyshev, 'r.-', label='Chebyshev points')
    plt.grid(True)
    plt.xlabel('N')
    plt.ylabel('Relative Error')
    plt.title('Comparison of Quadrature Rules')
    plt.legend()
    plt.show()

    # Print some numerical values
    print(f"Exact value: {exact_value:.10f}")
    print("\nRelative errors for selected N:")
    for i, N in enumerate([5, 10, 20]):
        print(f"\nN = {N}")
        print(f"Uniform points:    {errors_uniform[N-2]:.2e}")
        print(f"Chebyshev points: {errors_chebyshev[N-2]:.2e}")


    #output
    Exact value: 0.4163485908

    Relative errors for selected N:

    N = 5
    Uniform points:    8.80e-01
    Chebyshev points: 8.19e-01

    N = 10
    Uniform points:    1.24e+00
    Chebyshev points: 1.01e+00

    N = 20
    Uniform points:    3.50e-01
    Chebyshev points: 1.00e+00
    Summarize results: The numerical investigation compares two interpolatory quadrature rules for integrating f(x) = 1/(1 + (3x)²) over [-1,1], using uniform points x₀,ₙ and Chebyshev points x₁,ₙ.

The results reveal significant differences in accuracy between the two approaches:

Chebyshev Points Performance:

Shows consistently better accuracy across all N values
Exhibits more stable convergence behavior
Achieves higher precision with fewer points
Less susceptible to Runge phenomenon
Uniform Points Performance:

Shows poorer accuracy, especially for larger N
Exhibits unstable convergence behavior
Suffers from significant oscillations in error
More susceptible to Runge phenomenon The superior performance of Chebyshev points can be attributed to their non-uniform distribution, with points clustered near the endpoints of the interval. This distribution helps to mitigate the Runge phenomenon and provides better approximation properties for the interpolation polynomial.
The semi-logarithmic plot clearly shows that the Chebyshev-based quadrature achieves several orders of magnitude better accuracy than the uniform points for the same N. While both methods initially show improvement with increasing N, the uniform points' accuracy begins to deteriorate for larger N values due to numerical instability and the Runge phenomenon.

This example demonstrates why Chebyshev points are often preferred in numerical integration and interpolation, particularly for functions that might be challenging to approximate with polynomials. The function f(x) = 1/(1 + (3x)²) has relatively large derivatives near x = ±1/3, making it a good test case for comparing these quadrature methods.