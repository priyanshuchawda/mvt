#!/usr/bin/env python3
"""
Mean Value Theorem Calculator

This program computes the "mean value point" guaranteed by the Lagrange Mean Value Theorem (LMVT)
and, when applicable, the point(s) required by Rolle's Theorem (RMVT) for a user-specified function
over a given interval.

The program uses symbolic differentiation when possible and falls back to numerical methods when needed.
It validates the conditions of the theorems and provides clear output messages.

Enhanced with mpmath for high-precision arithmetic and improved numerical solvers.
"""

import sympy as sp
import numpy as np
import warnings
import mpmath as mp
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, diff, solve, sympify, nsolve
from sympy.utilities.lambdify import lambdify
from scipy import optimize

# Set the matplotlib backend to 'Agg' to avoid Tkinter-related issues
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt

# Set mpmath precision to a higher value for better accuracy with transcendental functions
mp.mp.dps = 50  # Decimal places of precision

# Suppress numpy warnings for invalid operations (like log of negative numbers)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Define the symbol for the variable
x = Symbol('x')

def is_continuous_on_interval(f, a, b):
    """
    Check if a function is continuous on the interval [a, b].
    This is a simplified check and may not catch all discontinuities.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        
    Returns:
        bool: True if the function appears continuous, False otherwise
    """
    try:
        # Convert symbolic function to a numerical function
        f_num = lambdify(x, f, 'numpy')
        
        # Check for undefined values at endpoints
        f_a = float(f.subs(x, a))
        f_b = float(f.subs(x, b))
        
        # Check for undefined values at some points in between
        test_points = np.linspace(a, b, 20)
        for point in test_points:
            f_val = float(f.subs(x, point))
            if np.isnan(f_val) or np.isinf(f_val):
                return False
        
        return True
    except Exception as e:
        print(f"Error checking continuity: {e}")
        return False

def is_differentiable_on_interval(f, a, b):
    """
    Check if a function is differentiable on the interval (a, b).
    This is a simplified check and may not catch all non-differentiable points.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        
    Returns:
        bool: True if the function appears differentiable, False otherwise
    """
    try:
        # Compute the derivative
        f_prime = diff(f, x)
        
        # Convert symbolic derivative to a numerical function
        f_prime_num = lambdify(x, f_prime, 'numpy')
        
        # Check for undefined derivative values at some points in the open interval
        test_points = np.linspace(a, b, 20)[1:-1]  # Exclude endpoints
        for point in test_points:
            f_prime_val = float(f_prime.subs(x, point))
            if np.isnan(f_prime_val) or np.isinf(f_prime_val):
                return False
        
        return True
    except Exception as e:
        print(f"Error checking differentiability: {e}")
        return False

def compute_lmvt(f, a, b, numerical_fallback=True):
    """
    Compute the point c in (a, b) where f'(c) equals the average slope,
    as guaranteed by the Lagrange Mean Value Theorem.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        numerical_fallback: Whether to use numerical methods if symbolic fails
        
    Returns:
        list: List of c values that satisfy the LMVT
        float: The average slope m
    """
    # Record the start time for performance tracking
    import time
    start_time = time.time()
    try:
        # Check if the interval is valid
        if a >= b:
            print("Error: The interval [a, b] must have a < b.")
            return [], None
        
        # Check if the function is continuous on [a, b] and differentiable on (a, b)
        if not is_continuous_on_interval(f, a, b):
            print("Error: The function must be continuous on the interval [a, b].")
            return [], None
        
        if not is_differentiable_on_interval(f, a, b):
            print("Error: The function must be differentiable on the interval (a, b).")
            return [], None
        
        # Calculate the average slope
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        m = (f_b - f_a) / (b - a)
        
        # Compute the derivative
        f_prime = diff(f, x)
        
        # Try symbolic solution first
        try:
            # Solve f'(c) = m
            equation = f_prime - m
            try:
                # Use a timeout mechanism that works across platforms
                import threading
                import time
                
                class SolveWithTimeout:
                    def __init__(self):
                        self.solutions = []
                        self.exception = None
                        self.completed = False
                    
                    def solve_func(self):
                        try:
                            self.solutions = solve(equation, x)
                            self.completed = True
                        except Exception as e:
                            self.exception = e
                            self.completed = True
                
                solver = SolveWithTimeout()
                solve_thread = threading.Thread(target=solver.solve_func)
                solve_thread.daemon = True
                solve_thread.start()
                
                # Wait for 3 seconds or until the solve function completes
                solve_thread.join(3)
                
                if not solver.completed:
                    print("Symbolic solution timed out, switching to numerical methods")
                    solutions = []
                elif solver.exception:
                    print(f"Symbolic solution failed: {solver.exception}")
                    solutions = []
                else:
                    solutions = solver.solutions
                
                # Filter solutions to keep only those in the open interval (a, b)
                c_values = [float(sol) for sol in solutions if sol.is_real and a < float(sol) < b]
                
                if c_values:
                    return c_values, float(m)
                # If no solutions found in the interval, continue to numerical methods
            except Exception as e:
                print(f"Error in symbolic solution process: {e}")
                # Continue to numerical methods
            
            if not numerical_fallback:
                return [], float(m)
        except KeyboardInterrupt:
            # Handle keyboard interrupt explicitly
            raise
        except Exception as e:
            print(f"Error in symbolic solution: {e}")
            # Continue to numerical methods
            if not numerical_fallback:
                return [], float(m)
        
        # Try SymPy's nsolve as a first numerical approach
        try:
            c_values = []
            # Try multiple starting points for nsolve
            test_points = np.linspace(a + 0.1*(b-a), b - 0.1*(b-a), 5)
            equation = f_prime - m
            
            for start_point in test_points:
                try:
                    # Use nsolve with the starting point
                    sol = float(nsolve(equation, x, start_point))
                    if a < sol < b:
                        c_values.append(sol)
                except Exception:
                    # If nsolve fails for this starting point, continue to the next
                    continue
            
            if c_values:
                # Remove duplicates and sort
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, float(m)
        except Exception as e:
            print(f"SymPy numerical solution failed: {e}")
            # Continue to SciPy methods
        
        # Fall back to SciPy numerical methods if SymPy methods fail
        try:
            # Define the function to find roots of: f'(x) - m
            def func_to_solve(val):
                return float(f_prime.subs(x, val)) - float(m)
            
            # Try to find solutions using sign changes in the interval
            c_values = []
            test_points = np.linspace(a, b, 30)[1:-1]  # Exclude endpoints, use more points
            
            # First try brentq method on subintervals where sign changes
            for i in range(len(test_points) - 1):
                left, right = test_points[i], test_points[i+1]
                try:
                    f_left = func_to_solve(left)
                    f_right = func_to_solve(right)
                    
                    # Check for sign change
                    if f_left * f_right <= 0:
                        try:
                            result = optimize.root_scalar(func_to_solve, bracket=[left, right], method='brentq')
                            if result.converged and a < result.root < b:
                                c_values.append(float(result.root))
                        except Exception:
                            # If brentq fails, try a general root finder
                            mid = (left + right) / 2
                            result = optimize.root(func_to_solve, mid)
                            if result.success and a < result.x[0] < b:
                                c_values.append(float(result.x[0]))
                except Exception:
                    continue
            
            # If no solutions found with sign changes, try multiple starting points
            if not c_values:
                for start_point in test_points:
                    try:
                        result = optimize.root(func_to_solve, start_point)
                        if result.success and a < result.x[0] < b:
                            c_values.append(float(result.x[0]))
                    except Exception:
                        continue
            
            # Remove duplicates and sort
            if c_values:
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, float(m)
        except Exception as e:
            print(f"SciPy numerical solution failed: {e}")
            
        # Final fallback: Try mpmath's findroot with high precision
        try:
            print("Attempting high-precision mpmath solution...")
            c_values = []
            
            # Convert the symbolic expression to an mpmath function
            def mpmath_func(val):
                try:
                    # Use mpmath's high precision
                    mp_val = mp.mpf(val)
                    result = float(f_prime.subs(x, mp_val)) - float(m)
                    return result
                except Exception:
                    return mp.inf  # Return infinity for invalid points
            
            # Try multiple starting points across the interval
            test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 10)
            for start in test_points:
                try:
                    # Use mpmath's findroot with high precision
                    result = mp.findroot(mpmath_func, start, solver='newton', tol=1e-20, maxsteps=100)
                    result_float = float(result)
                    if a < result_float < b:
                        c_values.append(result_float)
                except Exception:
                    continue
            
            # Remove duplicates and sort
            if c_values:
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, float(m)
                
        except Exception as e:
            print(f"mpmath solution failed: {e}")
    except Exception as e:
        print(f"Error in compute_lmvt: {e}")
    
    # If all methods fail, return empty list
    print("Could not find solutions. The function may be too complex or no solutions exist in the interval.")
    return [], float(m) if 'm' in locals() else None

def compute_rmvt(f, a, b, numerical_fallback=True):
    """
    Compute the point(s) c in (a, b) where f'(c) = 0,
    as required by Rolle's Theorem when f(a) = f(b).
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        numerical_fallback: Whether to use numerical methods if symbolic fails
        
    Returns:
        list: List of c values that satisfy Rolle's Theorem
        bool: Whether f(a) = f(b) (Rolle's condition)
    """
    try:
        # Check if the interval is valid
        if a >= b:
            print("Error: The interval [a, b] must have a < b.")
            return [], False
        
        # Check if f(a) = f(b), which is required for Rolle's Theorem
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        
        # Use a small epsilon for floating-point comparison
        epsilon = 1e-10
        rolles_condition = abs(float(f_a - f_b)) < epsilon
        
        if not rolles_condition:
            print("Note: Rolle's Theorem is not applicable because f(a) ≠ f(b).")
            return [], False
        
        # Check if the function is continuous on [a, b] and differentiable on (a, b)
        if not is_continuous_on_interval(f, a, b):
            print("Error: The function must be continuous on the interval [a, b].")
            return [], rolles_condition
        
        if not is_differentiable_on_interval(f, a, b):
            print("Error: The function must be differentiable on the interval (a, b).")
            return [], rolles_condition
        
        # Compute the derivative
        f_prime = diff(f, x)
        
        # Try symbolic solution first
        try:
            # Solve f'(c) = 0
            try:
                # Use a timeout mechanism that works across platforms
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
                
                # Wait for 3 seconds or until the solve function completes
                solve_thread.join(3)
                
                if not solver.completed:
                    print("Symbolic solution timed out, switching to numerical methods")
                    solutions = []
                elif solver.exception:
                    print(f"Symbolic solution failed: {solver.exception}")
                    solutions = []
                else:
                    solutions = solver.solutions
                
                # Filter solutions to keep only those in the open interval (a, b)
                c_values = [float(sol) for sol in solutions if sol.is_real and a < float(sol) < b]
                
                if c_values:
                    return c_values, rolles_condition
                # If no solutions found in the interval, continue to numerical methods
            except Exception as e:
                print(f"Error in symbolic solution process: {e}")
                # Continue to numerical methods
            
            if not numerical_fallback:
                return [], rolles_condition
        except KeyboardInterrupt:
            # Handle keyboard interrupt explicitly
            raise
        except Exception as e:
            print(f"Error in symbolic solution: {e}")
            # Continue to numerical methods
            if not numerical_fallback:
                return [], rolles_condition
        
        # Try SymPy's nsolve as a first numerical approach
        try:
            c_values = []
            # Try multiple starting points for nsolve
            test_points = np.linspace(a + 0.1*(b-a), b - 0.1*(b-a), 5)
            
            for start_point in test_points:
                try:
                    # Use nsolve with the starting point
                    sol = float(nsolve(f_prime, x, start_point))
                    if a < sol < b:
                        c_values.append(sol)
                except Exception:
                    # If nsolve fails for this starting point, continue to the next
                    continue
            
            if c_values:
                # Remove duplicates and sort
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, rolles_condition
        except Exception as e:
            print(f"SymPy numerical solution failed: {e}")
            # Continue to SciPy methods
        
        # Fall back to SciPy numerical methods if SymPy methods fail
        try:
            # Define the function to find roots of: f'(x)
            def func_to_solve(val):
                return float(f_prime.subs(x, val))
            
            # Try to find solutions using sign changes in the interval
            c_values = []
            test_points = np.linspace(a, b, 20)[1:-1]  # Exclude endpoints, use more points
            
            # First try brentq method on subintervals where sign changes
            for i in range(len(test_points) - 1):
                left, right = test_points[i], test_points[i+1]
                try:
                    f_left = func_to_solve(left)
                    f_right = func_to_solve(right)
                    
                    # Check for sign change
                    if f_left * f_right <= 0:
                        try:
                            result = optimize.root_scalar(func_to_solve, bracket=[left, right], method='brentq')
                            if result.converged and a < result.root < b:
                                c_values.append(float(result.root))
                        except Exception:
                            # If brentq fails, try a general root finder
                            mid = (left + right) / 2
                            result = optimize.root(func_to_solve, mid)
                            if result.success and a < result.x[0] < b:
                                c_values.append(float(result.x[0]))
                except Exception:
                    continue
            
            # If no solutions found with sign changes, try multiple starting points
            if not c_values:
                for start_point in test_points:
                    try:
                        result = optimize.root(func_to_solve, start_point)
                        if result.success and a < result.x[0] < b:
                            c_values.append(float(result.x[0]))
                    except Exception:
                        continue
            
            # Remove duplicates and sort
            if c_values:
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, rolles_condition
        except Exception as e:
            print(f"SciPy numerical solution failed: {e}")
            
        # Final fallback: Try mpmath's findroot with high precision
        try:
            print("Attempting high-precision mpmath solution for Rolle's Theorem...")
            c_values = []
            
            # Convert the symbolic expression to an mpmath function
            def mpmath_func(val):
                try:
                    # Use mpmath's high precision
                    mp_val = mp.mpf(val)
                    result = float(f_prime.subs(x, mp_val))
                    return result
                except Exception:
                    return mp.inf  # Return infinity for invalid points
            
            # Try multiple starting points across the interval
            test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 10)
            for start in test_points:
                try:
                    # Use mpmath's findroot with high precision
                    result = mp.findroot(mpmath_func, start, solver='newton', tol=1e-20, maxsteps=100)
                    result_float = float(result)
                    if a < result_float < b:
                        c_values.append(result_float)
                except Exception:
                    continue
            
            # Remove duplicates and sort
            if c_values:
                c_values = sorted(list(set([round(c, 10) for c in c_values])))
                return c_values, rolles_condition
                
        except Exception as e:
            print(f"mpmath solution failed: {e}")
    except Exception as e:
        print(f"Error in compute_rmvt: {e}")
    
    # If all methods fail, return empty list
    return [], rolles_condition

def plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope):
    """
    Plot the function, the secant line, and the points where LMVT and RMVT are satisfied.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        lmvt_points: List of points satisfying LMVT
        rmvt_points: List of points satisfying RMVT
        avg_slope: The average slope over the interval
    """
    try:
        # Convert symbolic function to a numerical function
        f_num = lambdify(x, f, 'numpy')
        
        # Create x values for plotting with error handling
        try:
            margin = (b - a) * 0.2
            x_vals = np.linspace(a - margin, b + margin, 1000)
            
            # Handle potential domain issues by filtering out points that cause errors
            def safe_eval(func, x_array):
                y_array = np.zeros_like(x_array)
                mask = np.ones_like(x_array, dtype=bool)
                for i, x_val in enumerate(x_array):
                    try:
                        y_array[i] = func(x_val)
                        if np.isnan(y_array[i]) or np.isinf(y_array[i]):
                            mask[i] = False
                    except Exception:
                        mask[i] = False
                return x_array[mask], y_array[mask]
            
            x_filtered, y_vals = safe_eval(f_num, x_vals)
            
            if len(x_filtered) < 10:
                # If too few valid points, try a more focused range
                x_vals = np.linspace(a, b, 1000)
                x_filtered, y_vals = safe_eval(f_num, x_vals)
                
            # Create the plot with a modern style
            plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Plot the function with improved styling
            plt.plot(x_filtered, y_vals, 'b-', linewidth=2.5, label=f'f(x) = {sp.latex(f)}')
            
            # Plot the interval endpoints with annotations
            f_a = float(f.subs(x, a))
            f_b = float(f.subs(x, b))
            plt.plot([a, b], [f_a, f_b], 'ro', markersize=8)
            plt.annotate(f'({a}, {f_a:.4f})', (a, f_a), textcoords="offset points", 
                         xytext=(-10,-15), ha='right')
            plt.annotate(f'({b}, {f_b:.4f})', (b, f_b), textcoords="offset points", 
                         xytext=(10,-15), ha='left')
            
            # Plot the secant line with improved styling
            secant_x = np.linspace(a, b, 100)
            secant_y = f_a + avg_slope * (secant_x - a)
            plt.plot(secant_x, secant_y, 'g--', linewidth=2, label=f'Secant line (slope = {avg_slope:.4f})')
            
            # Plot LMVT points with improved annotations
            for c in lmvt_points:
                f_c = float(f.subs(x, c))
                plt.plot(c, f_c, 'go', markersize=10)
                plt.annotate(f'LMVT: c = {c:.4f}', (c, f_c), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
                
                # Plot the tangent line at c with improved styling
                f_prime = diff(f, x)
                slope_at_c = float(f_prime.subs(x, c))
                tangent_x = np.linspace(c-0.2*(b-a), c+0.2*(b-a), 100)
                tangent_y = f_c + slope_at_c * (tangent_x - c)
                plt.plot(tangent_x, tangent_y, 'g-', linewidth=1.5, alpha=0.7)
                
                # Add a note about the derivative value
                plt.annotate(f"f'({c:.4f}) = {slope_at_c:.4f}", (c, f_c), 
                             textcoords="offset points", xytext=(0,30), 
                             ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            
            # Plot RMVT points with improved annotations
            for c in rmvt_points:
                f_c = float(f.subs(x, c))
                plt.plot(c, f_c, 'mo', markersize=10)
                plt.annotate(f'RMVT: c = {c:.4f}', (c, f_c), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))
                
                # Plot the horizontal tangent line at c with improved styling
                plt.plot([c-0.2*(b-a), c+0.2*(b-a)], [f_c, f_c], 'm-', linewidth=1.5, alpha=0.7)
                
                # Add a note that derivative is zero
                plt.annotate(f"f'({c:.4f}) = 0", (c, f_c), 
                             textcoords="offset points", xytext=(0,30), 
                             ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            
            # Add labels and title with improved styling
            plt.grid(True, alpha=0.4)
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title('Mean Value Theorem and Rolle\'s Theorem Visualization', fontsize=14, fontweight='bold')
            
            # Add a text box with theorem explanation
            theorem_text = "Lagrange MVT: If f is continuous on [a,b] and differentiable on (a,b),\n"
            theorem_text += f"then ∃c∈(a,b) such that f'(c) = {avg_slope:.4f} = [f(b)-f(a)]/(b-a)"
            plt.figtext(0.5, 0.01, theorem_text, ha="center", fontsize=10, 
                        bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Improve legend
            plt.legend(loc='best', framealpha=0.9, shadow=True)
            
            # Show the plot with tight layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for the theorem text
            plt.savefig('mvt_plot.png', dpi=300)
            plt.close()
            print("Enhanced plot saved as 'mvt_plot.png'")
            
        except Exception as e:
            print(f"Error generating plot data: {e}")
            # Fallback to a simpler plot if the enhanced one fails
            try:
                plt.figure(figsize=(10, 6))
                # Simple plot of function over the interval
                x_simple = np.linspace(a, b, 100)
                y_simple = [float(f.subs(x, xi)) for xi in x_simple]
                plt.plot(x_simple, y_simple, 'b-')
                plt.grid(True)
                plt.title(f"Function f(x) = {f} on [{a}, {b}]")
                plt.savefig('mvt_plot.png')
                plt.close()
                print("Simple fallback plot saved as 'mvt_plot.png'")
            except Exception:
                print("Could not generate even a simple plot. Skipping visualization.")
    except Exception as e:
        print(f"Error creating plot: {e}")