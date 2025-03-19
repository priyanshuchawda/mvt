#!/usr/bin/env python3
"""
Advanced Mean Value Theorem Calculator Extensions

This module provides enhanced numerical methods for the Mean Value Theorem calculator.
It includes improved fallback mechanisms, better handling of complex functions,
and smarter initial guessing for numerical solvers.

These functions extend the capabilities of the basic Mean Value Theorem calculator
to handle more complex functions and provide more robust solutions.
"""

import sympy as sp
import numpy as np
import mpmath as mp
import warnings
from sympy import Symbol, diff, solve, nsolve
from sympy.utilities.lambdify import lambdify
from scipy import optimize

# Set the matplotlib backend to 'Agg' to avoid Tkinter-related issues
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import make_interp_spline

# Set mpmath precision to a higher value for better accuracy with transcendental functions
mp.mp.dps = 100  # Increased decimal places of precision for more accurate results

# Suppress numpy warnings for invalid operations (like log of negative numbers)
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Suppress other numpy warnings that might occur
warnings.filterwarnings('ignore', category=Warning)

# Define the symbol for the variable
x = Symbol('x')

def adaptive_numerical_solver(f_prime, m, a, b, max_points=50, tol=1e-12):
    """
    Advanced numerical solver that adaptively increases the density of test points
    in regions where the function is changing rapidly.
    
    Args:
        f_prime: Symbolic derivative of the function
        m: Target value (average slope for LMVT)
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        max_points: Maximum number of initial test points
        tol: Tolerance for convergence
        
    Returns:
        list: List of c values that satisfy f'(c) = m
    """
    # Define the function to find roots of: f'(x) - m
    def func_to_solve(val):
        try:
            # Special handling for all fractional powers
            result = f_prime.subs(x, val)
            
            # Recursive function to handle fractional powers in complex expressions
            def process_pow_expression(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1:
                        # For any fractional power 1/n
                        n = exp.q
                        if n > 1:  # Only process non-integer powers
                            return sp.sign(base)**(1/n) * (sp.Abs(base) ** exp)
                
                # Process composite expressions recursively
                if expr.args:
                    new_args = [process_pow_expression(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            
            result = process_pow_expression(result)
            
            # Convert the result to float for numerical operations
            return float(result - m)
        except Exception as e:
            # For debugging, uncomment this:
            # print(f"Error evaluating at x={val}: {e}")
            return np.inf  # Return infinity for invalid points
    
    # Initial uniform sampling
    test_points = np.linspace(a, b, max_points)[1:-1]  # Exclude endpoints
    
    # Evaluate function at test points
    f_values = []
    valid_points = []
    for point in test_points:
        try:
            val = func_to_solve(point)
            if not np.isnan(val) and not np.isinf(val) and not isinstance(val, complex):
                f_values.append(val)
                valid_points.append(point)
        except Exception:
            continue
    
    if not valid_points:
        return []  # No valid points found
    
    # Convert to numpy arrays for easier manipulation
    valid_points = np.array(valid_points)
    f_values = np.array(f_values)
    
    # Find regions with sign changes (potential roots)
    c_values = []
    for i in range(len(valid_points) - 1):
        left, right = valid_points[i], valid_points[i+1]
        f_left, f_right = f_values[i], f_values[i+1]
        
        # Check for sign change or if one value is very close to zero
        if f_left * f_right <= 0 or abs(f_left) < tol or abs(f_right) < tol:
            # Refine this region with more points
            refined_points = np.linspace(left, right, 10)
            for j in range(len(refined_points) - 1):
                sub_left, sub_right = refined_points[j], refined_points[j+1]
                try:
                    f_sub_left = func_to_solve(sub_left)
                    f_sub_right = func_to_solve(sub_right)
                    
                    if f_sub_left * f_sub_right <= 0:
                        try:
                            # Try brentq method for precise root finding
                            result = optimize.root_scalar(func_to_solve, 
                                                          bracket=[sub_left, sub_right], 
                                                          method='brentq',
                                                          xtol=tol)
                            if result.converged and a < result.root < b:
                                c_values.append(float(result.root))
                        except Exception:
                            # If brentq fails, try a general root finder
                            mid = (sub_left + sub_right) / 2
                            result = optimize.root(func_to_solve, mid, tol=tol)
                            if result.success and a < result.x[0] < b:
                                c_values.append(float(result.x[0]))
                except Exception:
                    continue
    
    # If no solutions found with sign changes, try multiple starting points with different methods
    if not c_values:
        # Try different optimization methods
        methods = ['hybr', 'lm', 'krylov']
        for method in methods:
            for start_point in valid_points[::3]:  # Use every third point to reduce computation
                try:
                    result = optimize.root(func_to_solve, start_point, method=method, tol=tol)
                    if result.success and a < result.x[0] < b:
                        c_values.append(float(result.x[0]))
                except Exception:
                    continue
    
    # Remove duplicates and sort
    if c_values:
        # Use a more sophisticated duplicate removal that considers numerical precision
        unique_values = []
        for val in c_values:
            # Check if this value is already in our list (within tolerance)
            if not any(abs(val - existing) < tol for existing in unique_values):
                unique_values.append(val)
        
        return sorted(unique_values)
    
    return []

def multi_method_solver(f_prime, m, a, b):
    """
    Apply multiple numerical methods in parallel to find solutions.
    This increases the chance of finding all valid solutions.
    
    Args:
        f_prime: Symbolic derivative of the function
        m: Target value (average slope for LMVT)
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        
    Returns:
        list: List of c values that satisfy f'(c) = m
    """
    c_values = []
    
    # Define the function to find roots of: f'(x) - m
    def func_to_solve(val):
        try:
            return float(f_prime.subs(x, val)) - float(m)
        except Exception:
            return np.inf
    
    # Method 1: High-precision mpmath solution
    try:
        # Convert the symbolic expression to an mpmath function
        def mpmath_func(val):
            try:
                mp_val = mp.mpf(val)
                result = float(f_prime.subs(x, mp_val)) - float(m)
                return result
            except Exception:
                return mp.inf
        
        # Try multiple starting points and solvers
        test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 15)
        solvers = ['newton', 'secant', 'muller']
        
        for solver_method in solvers:
            for start in test_points:
                try:
                    result = mp.findroot(mpmath_func, start, solver=solver_method, 
                                         tol=1e-30, maxsteps=100)
                    result_float = float(result)
                    if a < result_float < b:
                        c_values.append(result_float)
                except Exception:
                    continue
    except Exception as e:
        print(f"mpmath solution attempt failed: {e}")
    
    # Method 2: Direct sampling for non-differentiable functions
    try:
        # Use a dense grid of points to evaluate the function directly
        # This is especially useful for non-differentiable functions like Abs(x)
        sample_points = np.linspace(a, b, 200)  # Use many points for better accuracy
        function_values = []
        valid_points = []
        
        for point in sample_points:
            try:
                val = func_to_solve(point)
                if not np.isnan(val) and not np.isinf(val):
                    function_values.append(val)
                    valid_points.append(point)
            except Exception:
                continue
        
        if valid_points:
            # Find points where the function is close to zero
            for i, point in enumerate(valid_points):
                if abs(function_values[i]) < 1e-8:
                    c_values.append(point)
            
            # Look for sign changes which indicate roots
            for i in range(len(valid_points) - 1):
                if function_values[i] * function_values[i+1] <= 0:
                    # Linear interpolation to estimate the root
                    x0, x1 = valid_points[i], valid_points[i+1]
                    y0, y1 = function_values[i], function_values[i+1]
                    if y0 != y1:  # Avoid division by zero
                        root = x0 - y0 * (x1 - x0) / (y1 - y0)
                        if a < root < b:
                            c_values.append(root)
    except Exception as e:
        print(f"Direct sampling method failed: {e}")
    
    # Method 3: Try SciPy's differential evolution only for well-behaved functions
    try:
        # Check if the function is well-behaved by sampling a few points
        test_ok = True
        test_samples = np.linspace(a, b, 10)
        for point in test_samples:
            try:
                val = func_to_solve(point)
                if np.isnan(val) or np.isinf(val):
                    test_ok = False
                    break
            except Exception:
                test_ok = False
                break
        
        if test_ok:
            bounds = [(a, b)]
            result = optimize.differential_evolution(
                lambda val: abs(func_to_solve(val[0])), 
                bounds, 
                maxiter=100,
                tol=1e-8,
                disp=False
            )
            if result.success and a < result.x[0] < b and abs(func_to_solve(result.x[0])) < 1e-8:
                c_values.append(float(result.x[0]))
    except Exception as e:
        print(f"Differential evolution attempt failed: {e}")
    
    # Method 4: Adaptive numerical solver
    try:
        adaptive_results = adaptive_numerical_solver(f_prime, m, a, b)
        c_values.extend(adaptive_results)
    except Exception as e:
        print(f"Adaptive solver attempt failed: {e}")
    
    # Remove duplicates and sort
    if c_values:
        # Use a more sophisticated duplicate removal
        tol = 1e-8
        unique_values = []
        for val in c_values:
            if not any(abs(val - existing) < tol for existing in unique_values):
                unique_values.append(val)
        
        return sorted(unique_values)
    
    return []

def enhanced_compute_lmvt(f, a, b, timeout=1):
    """
    Enhanced version of compute_lmvt with improved numerical methods.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        timeout: Maximum time in seconds to wait for symbolic solution
        
    Returns:
        list: List of c values that satisfy the LMVT
        float: The average slope m
    """
    try:
        # Check domain validity with enhanced warnings
        is_valid, invalid_points, domain_warnings = check_domain_validity(f, a, b)
        if domain_warnings:
            print("\nDomain Warnings:")
            for warning in domain_warnings:
                print(f"- {warning}")
        if not is_valid:
            print("\nFunction has invalid points in the interval:")
            print(f"Invalid at x = {invalid_points}")
            print("The calculator will attempt to work around these points.")
        
        # Calculate the average slope
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        m = (f_b - f_a) / (b - a)
        
        # Check if m is complex, if so, take the real part with warning
        if m.is_complex or (hasattr(m, 'has_nonzero_imaginary_part') and m.has_nonzero_imaginary_part()):
            print(f"Warning: Average slope contains complex values: {m}")
            print("Taking real part for calculations.")
            m = sp.re(m)
        
        # Compute the derivative
        f_prime = diff(f, x)
        print(f"Derivative: f'(x) = {f_prime}")
        
        # Try symbolic solution first with a timeout
        try:
            import threading
            import time
            
            class SolveWithTimeout:
                def __init__(self):
                    self.solutions = []
                    self.exception = None
                    self.completed = False
                
                def solve_func(self):
                    try:
                        equation = f_prime - m
                        self.solutions = solve(equation, x)
                        self.completed = True
                    except Exception as e:
                        self.exception = e
                        self.completed = True
            
            solver = SolveWithTimeout()
            solve_thread = threading.Thread(target=solver.solve_func)
            solve_thread.daemon = True
            solve_thread.start()
            
            # Wait for timeout seconds or until the solve function completes
            solve_thread.join(timeout)
            
            if not solver.completed:
                print("Symbolic solution timed out, switching to numerical methods")
            elif solver.exception:
                print(f"Symbolic solution failed: {solver.exception}")
            else:
                # Filter solutions to keep only those in the open interval (a, b)
                c_values = [float(sol) for sol in solver.solutions 
                           if sol.is_real and a < float(sol) < b]
                
                if c_values:
                    # Ensure m is a float to avoid complex conversion issues
                    return c_values, float(sp.re(m))
                else:
                    print("No symbolic solutions found in the interval, trying numerical methods")
        except Exception as e:
            print(f"Symbolic solution attempt failed: {e}")
        
        # If symbolic solution fails, use our enhanced numerical methods
        print("Attempting numerical solution...")
        c_values = multi_method_solver(f_prime, m, a, b)
        
        if c_values:
            # Ensure m is a float to avoid complex conversion issues
            return c_values, float(sp.re(m))
        else:
            # Final attempt with a very thorough search
            print("Attempting thorough numerical search...")
            # Increase the number of test points for a more thorough search
            c_values = adaptive_numerical_solver(f_prime, m, a, b, max_points=50, tol=1e-10)
            # Ensure m is a float to avoid complex conversion issues
            return c_values, float(sp.re(m))
            
    except Exception as e:
        print(f"Error in enhanced_compute_lmvt: {e}")
    
    # If all methods fail, return empty list and the slope if calculated
    if 'm' in locals():
        try:
            # Ensure we're returning a real float
            return [], float(sp.re(m))
        except Exception:
            return [], None
    else:
        return [], None

def enhanced_compute_rmvt(f, a, b, timeout=1):
    """
    Enhanced version of compute_rmvt with improved numerical methods.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        timeout: Maximum time in seconds to wait for symbolic solution
        
    Returns:
        list: List of c values that satisfy Rolle's Theorem
        bool: Whether f(a) = f(b) (Rolle's condition)
    """
    try:
        # Check if f(a) = f(b), which is required for Rolle's Theorem
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        
        # Use a small epsilon for floating-point comparison
        epsilon = 1e-10
        rolles_condition = abs(float(f_a - f_b)) < epsilon
        
        if not rolles_condition:
            return [], False
        
        # Compute the derivative
        f_prime = diff(f, x)
        print(f"Derivative for Rolle's Theorem: f'(x) = {f_prime}")
        
        # Try symbolic solution first with a timeout
        try:
            import threading
            import time
            
            class SolveWithTimeout:
                def __init__(self):
                    self.solutions = []
                    self.exception = None
                    self.completed = False
                
                def solve_func(self):
                    try:
                        self.solutions = solve(f_prime, x)
                        self.completed = True
                    except Exception as e:
                        self.exception = e
                        self.completed = True
            
            solver = SolveWithTimeout()
            solve_thread = threading.Thread(target=solver.solve_func)
            solve_thread.daemon = True
            solve_thread.start()
            
            # Wait for timeout seconds or until the solve function completes
            solve_thread.join(timeout)
            
            if not solver.completed:
                print("Symbolic solution timed out, switching to numerical methods")
            elif solver.exception:
                print(f"Symbolic solution failed: {solver.exception}")
            else:
                # Filter solutions to keep only those in the open interval (a, b)
                c_values = [float(sol) for sol in solver.solutions 
                           if sol.is_real and a < float(sol) < b]
                
                if c_values:
                    return c_values, rolles_condition
                else:
                    print("No symbolic solutions found in the interval, trying numerical methods")
        except Exception as e:
            print(f"Symbolic solution attempt failed: {e}")
        
        # If symbolic solution fails, use our enhanced numerical methods
        print("Attempting numerical solution for Rolle's Theorem...")
        c_values = multi_method_solver(f_prime, 0, a, b)  # For Rolle's, m = 0
        
        if c_values:
            return c_values, rolles_condition
        else:
            # Final attempt with a very thorough search
            print("Attempting thorough numerical search for Rolle's Theorem...")
            # Increase the number of test points for a more thorough search
            c_values = adaptive_numerical_solver(f_prime, 0, a, b, max_points=50, tol=1e-10)
            return c_values, rolles_condition
            
    except Exception as e:
        print(f"Error in enhanced_compute_rmvt: {e}")
    
    # If all methods fail, return empty list and the condition
    return [], rolles_condition

def check_domain_validity(f, a, b, num_points=20):
    """
    Check if a function is valid over a domain by sampling points.
    Now includes better handling of complex results and special cases.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        num_points: Number of test points to sample
        
    Returns:
        tuple: (is_valid, invalid_points, warnings) where:
               - is_valid is a boolean
               - invalid_points is a list of x values where the function is undefined
               - warnings is a list of strings with specific warnings about the domain
    """
    test_points = np.linspace(a, b, num_points)
    invalid_points = []
    warnings = []
    
    # First check if the function involves potential complex outputs
    expr_str = str(f)
    # Look for any fractional powers using a more general regex pattern
    import re
    fractional_power_pattern = r'\*\*\s*\(\s*1\s*/\s*(\d+)\s*\)'
    root_matches = re.findall(fractional_power_pattern, expr_str)
    
    if root_matches:
        for n in root_matches:
            if int(n) % 2 == 0:  # Even roots
                warnings.append(f"Function contains {n}th roots which may produce complex values for negative inputs.")
            else:  # Odd roots
                warnings.append(f"Function contains {n}th roots which may need special handling for negative inputs.")
    elif ']**(1/3)' in expr_str or '**(1/3)' in expr_str or 'cbrt' in expr_str:
        warnings.append("Function contains cube roots which may need special handling for negative inputs.")
    elif ']**(1/2)' in expr_str or '**(1/2)' in expr_str or 'sqrt' in expr_str:
        warnings.append("Function contains square roots which may produce complex values for negative inputs.")
    
    # Check points for validity using our custom eval approach
    def safe_eval(expr, point):
        try:
            # First use sympy's evalf for exact symbolic evaluation
            result = expr.subs(x, point)
            
            # Handle fractional powers of negative values
            def process_pow_expression(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q > 1:
                        n = exp.q
                        if n > 1:  # Only process non-integer powers
                            return sp.sign(base)**(1/n) * (sp.Abs(base) ** exp)
                
                # Process composite expressions recursively
                if expr.args:
                    new_args = [process_pow_expression(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            
            # If symbolic eval results in complex, try our custom approach
            if 'I' in str(result):
                result = process_pow_expression(expr).subs(x, point)
            
            # Convert to float for numerical checks
            float_result = float(result)
            return float_result, np.isnan(float_result) or np.isinf(float_result)
        except Exception as e:
            return None, True
    
    for point in test_points:
        result, is_invalid = safe_eval(f, point)
        if is_invalid:
            invalid_points.append(point)
            if result is None:
                # Try to identify the specific issue
                try:
                    f.subs(x, point)
                except Exception as e:
                    err_msg = str(e)
                    if "cannot convert complex" in err_msg:
                        warnings.append(f"Complex values detected near x = {point}")
                    elif "division by zero" in err_msg:
                        warnings.append(f"Division by zero detected near x = {point}")
    
    # Check for removable discontinuities
    if '/' in str(f):
        warnings.append("Function contains division. Check for removable discontinuities.")
    
    return len(invalid_points) == 0, invalid_points, warnings

def enhanced_plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope):
    """
    Create an enhanced visualization of the function and the Mean Value Theorem points.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        lmvt_points: List of points satisfying LMVT
        rmvt_points: List of points satisfying RMVT
        avg_slope: The average slope over the interval
    """
    # Convert symbolic function to a numerical function
    f_num = lambdify(x, f, 'numpy')
    
    # Create a smooth curve for plotting
    padding = (b - a) * 0.2  # Add padding to the left and right of the interval
    x_min, x_max = a - padding, b + padding
    
    # Use more points for a smoother curve
    x_vals = np.linspace(x_min, x_max, 1000)
    
    # Handle potential evaluation errors
    y_vals = []
    valid_x = []
    for x_val in x_vals:
        try:
            y_val = float(f.subs(x, x_val))
            if not np.isnan(y_val) and not np.isinf(y_val):
                y_vals.append(y_val)
                valid_x.append(x_val)
        except Exception:
            continue
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    if len(valid_x) > 3:  # Need at least 3 points for spline
        try:
            # Try to create a smooth spline for better visualization
            spl = make_interp_spline(valid_x, y_vals, k=3)
            smooth_x = np.linspace(min(valid_x), max(valid_x), 1000)
            smooth_y = spl(smooth_x)
            ax.plot(smooth_x, smooth_y, 'b-', linewidth=2, label=f'f(x) = {f}')
        except Exception:
            # Fall back to regular plot if spline fails
            ax.plot(valid_x, y_vals, 'b-', linewidth=2, label=f'f(x) = {f}')
    else:
        # Not enough points for spline, use regular plot
        ax.plot(valid_x, y_vals, 'b-', linewidth=2, label=f'f(x) = {f}')
    
    # Calculate function values at endpoints
    f_a = float(f.subs(x, a))
    f_b = float(f.subs(x, b))
    
    # Plot the secant line connecting (a, f(a)) and (b, f(b))
    ax.plot([a, b], [f_a, f_b], 'r--', linewidth=1.5, label=f'Secant line (slope = {avg_slope:.4f})')
    
    # Highlight the interval [a, b]
    ax.axvline(x=a, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=b, color='gray', linestyle=':', alpha=0.7)
    
    # Mark the endpoints
    ax.plot(a, f_a, 'ro', markersize=6)
    ax.plot(b, f_b, 'ro', markersize=6)
    ax.annotate(f'({a}, {f_a:.4f})', (a, f_a), xytext=(-10, -20), 
                textcoords='offset points', ha='right')
    ax.annotate(f'({b}, {f_b:.4f})', (b, f_b), xytext=(10, -20), 
                textcoords='offset points', ha='left')
    
    # Shade the region under the curve
    try:
        # Create a polygon to shade the area under the curve
        x_interval = np.linspace(a, b, 100)
        y_interval = [float(f.subs(x, xi)) for xi in x_interval]
        verts = [(a, 0)] + list(zip(x_interval, y_interval)) + [(b, 0)]
        poly = Polygon(verts, facecolor='lightblue', edgecolor='none', alpha=0.3)
        ax.add_patch(poly)
    except Exception:
        pass  # Skip shading if there's an error
    
    # Plot LMVT points
    for c in lmvt_points:
        f_c = float(f.subs(x, c))
        # Calculate the tangent line at point c
        f_prime = sp.diff(f, x)
        slope = float(f_prime.subs(x, c))
        
        # Plot the tangent line
        tangent_x = np.linspace(c - (b-a)/4, c + (b-a)/4, 2)
        tangent_y = [f_c + slope * (xx - c) for xx in tangent_x]
        ax.plot(tangent_x, tangent_y, 'g-', linewidth=1.5)
        
        # Mark the point
        ax.plot(c, f_c, 'go', markersize=8)
        ax.annotate(f'LMVT: ({c:.4f}, {f_c:.4f})', (c, f_c), xytext=(0, 20), 
                    textcoords='offset points', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # Plot RMVT points
    for c in rmvt_points:
        f_c = float(f.subs(x, c))
        # Calculate the tangent line at point c (horizontal for RMVT)
        tangent_x = np.linspace(c - (b-a)/4, c + (b-a)/4, 2)
        tangent_y = [f_c, f_c]  # Horizontal line since derivative is 0
        ax.plot(tangent_x, tangent_y, 'm-', linewidth=1.5)
        
        # Mark the point
        ax.plot(c, f_c, 'mo', markersize=8)
        ax.annotate(f'RMVT: ({c:.4f}, {f_c:.4f})', (c, f_c), xytext=(0, -20), 
                    textcoords='offset points', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3))
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Mean Value Theorem Visualization')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Adjust the view to focus on the relevant part
    y_vals_interval = [float(f.subs(x, xi)) for xi in np.linspace(a, b, 20) if not np.isnan(float(f.subs(x, xi)))]
    if y_vals_interval:
        y_min, y_max = min(y_vals_interval), max(y_vals_interval)
        y_padding = (y_max - y_min) * 0.3
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('mvt_plot.png', dpi=300)
    plt.close()
    
    print("\nVisualization saved as 'mvt_plot.png'")

def integrate_with_calculator(main_module=None):
    """
    Integrate the advanced MVT functions with the main calculator.
    This function can be called from the main module to use the enhanced functions.
    
    Args:
        main_module: The main module to integrate with (optional)
    
    Returns:
        dict: A dictionary of enhanced functions that can be used by the main calculator
    """
    enhanced_functions = {
        'compute_lmvt': enhanced_compute_lmvt,
        'compute_rmvt': enhanced_compute_rmvt,
        'plot_function_and_theorems': enhanced_plot_function_and_theorems,
        'check_domain_validity': check_domain_validity
    }
    
    if main_module:
        # If a module is provided, we can directly modify its attributes
        for name, func in enhanced_functions.items():
            setattr(main_module, name, func)
        print("Enhanced MVT functions integrated with the main calculator.")
    
    return enhanced_functions

# If this module is run directly, provide a simple demonstration
if __name__ == "__main__":
    print("Advanced Mean Value Theorem Calculator Extensions")
    print("This module provides enhanced numerical methods for the MVT calculator.")
    print("Import this module in the main calculator to use the enhanced functions.")
    
    # Simple demonstration
    try:
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        
        # Set up transformations for more user-friendly input
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        # Example function and interval
        func_str = "x**3 - 2*x**2 + 2"
        f = parse_expr(func_str, transformations=transformations)
        a, b = 0, 2
        
        print(f"\nExample: f(x) = {f}, interval [{a}, {b}]")
        
        # Check domain validity
        is_valid, invalid_points = check_domain_validity(f, a, b)
        if is_valid:
            print("Function is valid over the entire interval.")
        else:
            print(f"Warning: Function is undefined at points: {invalid_points}")
        
        # Compute LMVT
        lmvt_points, avg_slope = enhanced_compute_lmvt(f, a, b)
        print(f"\nLMVT Results:")
        print(f"Average slope: {avg_slope}")
        print(f"Points satisfying LMVT: {lmvt_points}")
        
        # Compute RMVT
        rmvt_points, rolles_condition = enhanced_compute_rmvt(f, a, b)
        print(f"\nRMVT Results:")
        print(f"Rolle's condition satisfied: {rolles_condition}")
        print(f"Points satisfying RMVT: {rmvt_points}")
        
        # Create visualization
        enhanced_plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope)
        
    except Exception as e:
        print(f"Error in demonstration: {e}")