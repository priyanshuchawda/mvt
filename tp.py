#!/usr/bin/env python3
"""
Merged Mean Value Theorem Calculator

This single file combines:
  - mean_value_theorem.py (basic functions for LMVT/Rolle's Theorem),
  - mvt_advanced.py (enhanced numerical methods), and
  - mvt_calculator.py (the main program interface).

Usage:
    python merged_mvt_calculator.py
"""

import sympy as sp
import numpy as np
import mpmath as mp
import warnings
import re
import threading
import time

# Set matplotlib to use non-interactive Agg backend to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import optimize
from scipy.interpolate import make_interp_spline

# Increase mpmath precision
mp.mp.dps = 100

# Suppress warnings from numpy and others
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=Warning)

# Define the symbol for variable x (global)
x = sp.Symbol('x')

#######################################
#          Basic Functions            #
#######################################

def handle_removable_discontinuities(f, x):
    """
    Simplify expressions with removable discontinuities.
    """
    try:
        simplified = sp.simplify(f)
        if '/' in str(simplified):
            num, den = simplified.as_numer_denom()
            cancelled = sp.cancel(simplified)
            if len(str(cancelled)) < len(str(simplified)):
                return cancelled
        return simplified
    except Exception as e:
        print(f"Warning: Could not simplify expression: {e}")
        return f

def is_continuous_on_interval(f, a, b):
    """
    Check if function f is continuous on [a, b].
    """
    try:
        f_simplified = handle_removable_discontinuities(f, x)
        f_a = float(f_simplified.subs(x, a))
        f_b = float(f_simplified.subs(x, b))
        test_points = np.linspace(a, b, 20)
        for point in test_points:
            f_val = float(f_simplified.subs(x, point))
            if np.isnan(f_val) or np.isinf(f_val):
                return False
        return True
    except Exception as e:
        print(f"Error checking continuity: {e}")
        return False

def custom_derivative(f, x):
    """
    Compute derivative with special handling for fractional powers and absolute values.
    """
    try:
        if isinstance(f, sp.Pow):
            base, exp = f.args
            if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q > 1:
                n = exp.q
                f = sp.sign(base) * (sp.Abs(base) ** exp)
                return sp.sign(base) / (n * (sp.Abs(base) ** (sp.Rational(n-1, n))))
        elif isinstance(f, sp.Abs):
            return sp.Piecewise((1, x > 0), (-1, x < 0), (sp.nan, True))
        elif hasattr(f, 'args') and f.args:
            needs_special = any(isinstance(arg, sp.Pow) and isinstance(arg.args[1], sp.Rational) and arg.args[1].p == 1 and arg.args[1].q > 1 for arg in f.args)
            if needs_special:
                if f.func == sp.Add:
                    return sp.Add(*[custom_derivative(arg, x) for arg in f.args])
                elif f.func == sp.Mul:
                    u, v = f.args[0], sp.Mul(*f.args[1:]) if len(f.args) > 1 else 1
                    return u * custom_derivative(v, x) + custom_derivative(u, x) * v
                elif f.func == sp.Pow:
                    base, power = f.args
                    if power.has(x):
                        return f * (custom_derivative(power, x) * sp.log(base) + power * custom_derivative(base, x) / base)
                    else:
                        return power * base**(power-1) * custom_derivative(base, x)
        return sp.diff(f, x)
    except Exception as e:
        print(f"Error in custom derivative: {e}")
        return sp.diff(f, x)

def is_differentiable_on_interval(f, a, b):
    """
    Check if function f is differentiable on (a, b).
    """
    try:
        f_prime = custom_derivative(f, x)
        test_points = np.linspace(a, b, 20)[1:-1]
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
    Compute the point(s) c in (a, b) where f'(c) equals the average slope.
    """
    start_time = time.time()
    try:
        if a >= b:
            print("Error: a must be less than b.")
            return [], None
        f = handle_removable_discontinuities(f, x)
        print(f"Simplified function: f(x) = {f}")
        if not is_continuous_on_interval(f, a, b):
            print("Error: Function not continuous on [a, b].")
            return [], None
        if not is_differentiable_on_interval(f, a, b):
            print("Error: Function not differentiable on (a, b).")
            return [], None
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        m = (f_b - f_a) / (b - a)
        f_prime = custom_derivative(f, x)
        print(f"Computed derivative: f'(x) = {f_prime}")
        # Try symbolic solution with timeout
        class SolveWithTimeout:
            def __init__(self):
                self.solutions = []
                self.exception = None
                self.completed = False
            def solve_func(self):
                try:
                    self.solutions = sp.solve(f_prime - m, x)
                    self.completed = True
                except Exception as e:
                    self.exception = e
                    self.completed = True
        solver = SolveWithTimeout()
        thread = threading.Thread(target=solver.solve_func)
        thread.daemon = True
        thread.start()
        thread.join(3)
        if solver.completed and solver.exception is None:
            solutions = solver.solutions
            c_values = [float(sol) for sol in solutions if sol.is_real and a < float(sol) < b]
            if c_values:
                return c_values, float(m)
        if not numerical_fallback:
            return [], float(m)
        # Try SymPy nsolve method
        c_values = []
        test_points = np.linspace(a + 0.1*(b-a), b - 0.1*(b-a), 5)
        for start_point in test_points:
            try:
                sol = float(sp.nsolve(f_prime - m, x, start_point))
                if a < sol < b:
                    c_values.append(sol)
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, float(m)
        # Fall back to SciPy numerical methods
        def func_to_solve(val):
            return float(f_prime.subs(x, val)) - float(m)
        c_values = []
        test_points = np.linspace(a, b, 30)[1:-1]
        for i in range(len(test_points) - 1):
            left, right = test_points[i], test_points[i+1]
            try:
                f_left = func_to_solve(left)
                f_right = func_to_solve(right)
                if f_left * f_right <= 0:
                    result = optimize.root_scalar(func_to_solve, bracket=[left, right], method='brentq')
                    if result.converged and a < result.root < b:
                        c_values.append(float(result.root))
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, float(m)
        # Final fallback: mpmath
        print("Attempting high-precision mpmath solution...")
        c_values = []
        def mpmath_func(val):
            try:
                mp_val = mp.mpf(val)
                return float(f_prime.subs(x, mp_val)) - float(m)
            except Exception:
                return mp.inf
        test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 10)
        for start in test_points:
            try:
                result = mp.findroot(mpmath_func, start, solver='newton', tol=1e-20, maxsteps=100)
                result_float = float(result)
                if a < result_float < b:
                    c_values.append(result_float)
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, float(m)
    except Exception as e:
        print(f"Error in compute_lmvt: {e}")
    print("Could not find solutions for LMVT.")
    return [], float(m) if 'm' in locals() else None

def compute_rmvt(f, a, b, numerical_fallback=True):
    """
    Compute the point(s) c in (a, b) where f'(c)=0 if f(a)=f(b) (Rolle's Theorem).
    """
    try:
        if a >= b:
            print("Error: a must be less than b.")
            return [], False
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        epsilon = 1e-10
        rolles_condition = abs(float(f_a - f_b)) < epsilon
        if not rolles_condition:
            print("Note: Rolle's Theorem not applicable because f(a) ≠ f(b).")
            return [], False
        if not is_continuous_on_interval(f, a, b):
            print("Error: Function not continuous on [a, b].")
            return [], rolles_condition
        if not is_differentiable_on_interval(f, a, b):
            print("Error: Function not differentiable on (a, b).")
            return [], rolles_condition
        f_prime = custom_derivative(f, x)
        class SolveWithTimeout:
            def __init__(self):
                self.solutions = []
                self.exception = None
                self.completed = False
            def solve_func(self):
                try:
                    self.solutions = sp.solve(f_prime, x)
                    self.completed = True
                except Exception as e:
                    self.exception = e
                    self.completed = True
        solver = SolveWithTimeout()
        thread = threading.Thread(target=solver.solve_func)
        thread.daemon = True
        thread.start()
        thread.join(3)
        if solver.completed and solver.exception is None:
            solutions = solver.solutions
            c_values = [float(sol) for sol in solutions if sol.is_real and a < float(sol) < b]
            if c_values:
                return c_values, rolles_condition
        if not numerical_fallback:
            return [], rolles_condition
        c_values = []
        test_points = np.linspace(a + 0.1*(b-a), b - 0.1*(b-a), 5)
        for start_point in test_points:
            try:
                sol = float(sp.nsolve(f_prime, x, start_point))
                if a < sol < b:
                    c_values.append(sol)
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, rolles_condition
        def func_to_solve(val):
            return float(f_prime.subs(x, val))
        c_values = []
        test_points = np.linspace(a, b, 20)[1:-1]
        for i in range(len(test_points) - 1):
            left, right = test_points[i], test_points[i+1]
            try:
                if func_to_solve(left) * func_to_solve(right) <= 0:
                    result = optimize.root_scalar(func_to_solve, bracket=[left, right], method='brentq')
                    if result.converged and a < result.root < b:
                        c_values.append(float(result.root))
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, rolles_condition
        print("Attempting high-precision mpmath solution for Rolle's Theorem...")
        c_values = []
        def mpmath_func(val):
            try:
                mp_val = mp.mpf(val)
                return float(f_prime.subs(x, mp_val))
            except Exception:
                return mp.inf
        test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 10)
        for start in test_points:
            try:
                result = mp.findroot(mpmath_func, start, solver='newton', tol=1e-20, maxsteps=100)
                result_float = float(result)
                if a < result_float < b:
                    c_values.append(result_float)
            except Exception:
                continue
        if c_values:
            c_values = sorted(list(set([round(c, 10) for c in c_values])))
            return c_values, rolles_condition
    except Exception as e:
        print(f"Error in compute_rmvt: {e}")
    return [], rolles_condition

def plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope):
    """
    Plot the function, secant line, and LMVT/RMVT points.
    """
    f_num = sp.lambdify(x, f, 'numpy')
    margin = (b - a) * 0.2
    x_vals = np.linspace(a - margin, b + margin, 1000)
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
        x_vals = np.linspace(a, b, 1000)
        x_filtered, y_vals = safe_eval(f_num, x_vals)
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(x_filtered, y_vals, 'b-', linewidth=2.5, label=f'f(x) = {sp.latex(f)}')
    f_a = float(f.subs(x, a))
    f_b = float(f.subs(x, b))
    plt.plot([a, b], [f_a, f_b], 'ro', markersize=8)
    plt.annotate(f'({a}, {f_a:.4f})', (a, f_a), textcoords="offset points", xytext=(-10,-15), ha='right')
    plt.annotate(f'({b}, {f_b:.4f})', (b, f_b), textcoords="offset points", xytext=(10,-15), ha='left')
    secant_x = np.linspace(a, b, 100)
    secant_y = f_a + avg_slope*(secant_x - a)
    plt.plot(secant_x, secant_y, 'r--', linewidth=1.5, label=f'Secant line (slope = {avg_slope:.4f})')
    for c in lmvt_points:
        f_c = float(f.subs(x, c))
        f_prime = sp.diff(f, x)
        slope = float(f_prime.subs(x, c))
        tangent_x = np.linspace(c - (b-a)/4, c + (b-a)/4, 2)
        tangent_y = [f_c + slope * (xx - c) for xx in tangent_x]
        plt.plot(tangent_x, tangent_y, 'g-', linewidth=1.5)
        plt.plot(c, f_c, 'go', markersize=8)
        plt.annotate(f'LMVT: ({c:.4f}, {f_c:.4f})', (c, f_c), xytext=(0, 20), textcoords='offset points', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    plt.title("Mean Value Theorem Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

#######################################
#        Advanced Functions           #
#######################################

def adaptive_numerical_solver(f_prime, m, a, b, max_points=50, tol=1e-12):
    """
    Adaptive numerical solver that increases density of test points.
    """
    def func_to_solve(val):
        try:
            result = f_prime.subs(x, val)
            def process_pow_expression(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1:
                        n = exp.q
                        if n > 1:
                            return sp.sign(base)**(1/n) * (sp.Abs(base) ** exp)
                if expr.args:
                    new_args = [process_pow_expression(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            result = process_pow_expression(result)
            return float(result - m)
        except Exception:
            return np.inf
    test_points = np.linspace(a, b, max_points)[1:-1]
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
        return []
    valid_points = np.array(valid_points)
    f_values = np.array(f_values)
    c_values = []
    for i in range(len(valid_points) - 1):
        left, right = valid_points[i], valid_points[i+1]
        f_left, f_right = f_values[i], f_values[i+1]
        if f_left * f_right <= 0 or abs(f_left) < tol or abs(f_right) < tol:
            refined_points = np.linspace(left, right, 10)
            for j in range(len(refined_points) - 1):
                sub_left, sub_right = refined_points[j], refined_points[j+1]
                try:
                    f_sub_left = func_to_solve(sub_left)
                    f_sub_right = func_to_solve(sub_right)
                    if f_sub_left * f_sub_right <= 0:
                        try:
                            result = optimize.root_scalar(func_to_solve, bracket=[sub_left, sub_right], method='brentq', xtol=tol)
                            if result.converged and a < result.root < b:
                                c_values.append(float(result.root))
                        except Exception:
                            mid = (sub_left + sub_right) / 2
                            result = optimize.root(func_to_solve, mid, tol=tol)
                            if result.success and a < result.x[0] < b:
                                c_values.append(float(result.x[0]))
                except Exception:
                    continue
    if not c_values:
        methods = ['hybr', 'lm', 'krylov']
        for method in methods:
            for start_point in valid_points[::3]:
                try:
                    result = optimize.root(func_to_solve, start_point, method=method, tol=tol)
                    if result.success and a < result.x[0] < b:
                        c_values.append(float(result.x[0]))
                except Exception:
                    continue
    if c_values:
        unique_values = []
        for val in c_values:
            if not any(abs(val - existing) < tol for existing in unique_values):
                unique_values.append(val)
        return sorted(unique_values)
    return []

def multi_method_solver(f_prime, m, a, b):
    """
    Apply multiple numerical methods to find solutions.
    """
    c_values = []
    def func_to_solve(val):
        try:
            return float(f_prime.subs(x, val)) - float(m)
        except Exception:
            return np.inf
    # Method 1: High-precision mpmath
    try:
        def mpmath_func(val):
            try:
                mp_val = mp.mpf(val)
                return float(f_prime.subs(x, mp_val)) - float(m)
            except Exception:
                return mp.inf
        test_points = np.linspace(a + 0.05*(b-a), b - 0.05*(b-a), 15)
        solvers = ['newton', 'secant', 'muller']
        for solver_method in solvers:
            for start in test_points:
                try:
                    result = mp.findroot(mpmath_func, start, solver=solver_method, tol=1e-30, maxsteps=100)
                    result_float = float(result)
                    if a < result_float < b:
                        c_values.append(result_float)
                except Exception:
                    continue
    except Exception as e:
        print(f"mpmath solution attempt failed: {e}")
    # Method 2: Direct sampling
    try:
        sample_points = np.linspace(a, b, 200)
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
            for i, point in enumerate(valid_points):
                if abs(function_values[i]) < 1e-8:
                    c_values.append(point)
            for i in range(len(valid_points) - 1):
                if function_values[i] * function_values[i+1] <= 0:
                    x0, x1 = valid_points[i], valid_points[i+1]
                    y0, y1 = function_values[i], function_values[i+1]
                    if y0 != y1:
                        root = x0 - y0 * (x1 - x0) / (y1 - y0)
                        if a < root < b:
                            c_values.append(root)
    except Exception as e:
        print(f"Direct sampling method failed: {e}")
    # Method 3: SciPy's differential evolution
    try:
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
            result = optimize.differential_evolution(lambda val: abs(func_to_solve(val[0])), bounds, maxiter=100, tol=1e-8, disp=False)
            if result.success and a < result.x[0] < b and abs(func_to_solve(result.x[0])) < 1e-8:
                c_values.append(float(result.x[0]))
    except Exception as e:
        print(f"Differential evolution attempt failed: {e}")
    # Method 4: Adaptive solver
    try:
        adaptive_results = adaptive_numerical_solver(f_prime, m, a, b)
        c_values.extend(adaptive_results)
    except Exception as e:
        print(f"Adaptive solver attempt failed: {e}")
    if c_values:
        tol_local = 1e-8
        unique_values = []
        for val in c_values:
            if not any(abs(val - existing) < tol_local for existing in unique_values):
                unique_values.append(val)
        return sorted(unique_values)
    return []

def enhanced_compute_lmvt(f, a, b, timeout=1):
    """
    Enhanced version of compute_lmvt with improved numerical methods.
    """
    try:
        is_valid, invalid_points, domain_warnings = check_domain_validity(f, a, b)
        if domain_warnings:
            print("\nDomain Warnings:")
            for warning in domain_warnings:
                print(f"- {warning}")
        if not is_valid:
            print("\nFunction has invalid points in the interval:")
            print(f"Invalid at x = {invalid_points}")
            print("Attempting to work around these points.")
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        m = (f_b - f_a) / (b - a)
        if m.is_complex or (hasattr(m, 'has_nonzero_imaginary_part') and m.has_nonzero_imaginary_part()):
            print(f"Warning: Average slope contains complex values: {m}")
            print("Taking real part for calculations.")
            m = sp.re(m)
        f_prime = sp.diff(f, x)
        print(f"Derivative: f'(x) = {f_prime}")
        class SolveWithTimeout:
            def __init__(self):
                self.solutions = []
                self.exception = None
                self.completed = False
            def solve_func(self):
                try:
                    equation = f_prime - m
                    self.solutions = sp.solve(equation, x)
                    self.completed = True
                except Exception as e:
                    self.exception = e
                    self.completed = True
        solver = SolveWithTimeout()
        thread = threading.Thread(target=solver.solve_func)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if solver.completed and solver.exception is None:
            c_values = [float(sol) for sol in solver.solutions if sol.is_real and a < float(sol) < b]
            if c_values:
                return c_values, float(sp.re(m))
            else:
                print("No symbolic solutions found; trying numerical methods")
        else:
            print("Symbolic solution timed out or failed; switching to numerical methods")
        print("Attempting numerical solution...")
        c_values = multi_method_solver(f_prime, m, a, b)
        if c_values:
            return c_values, float(sp.re(m))
        else:
            print("Attempting thorough numerical search...")
            c_values = adaptive_numerical_solver(f_prime, m, a, b, max_points=50, tol=1e-10)
            return c_values, float(sp.re(m))
    except Exception as e:
        print(f"Error in enhanced_compute_lmvt: {e}")
    if 'm' in locals():
        try:
            return [], float(sp.re(m))
        except Exception:
            return [], None
    else:
        return [], None

def enhanced_compute_rmvt(f, a, b, timeout=1):
    """
    Enhanced version of compute_rmvt with improved numerical methods.
    """
    try:
        f_a = f.subs(x, a)
        f_b = f.subs(x, b)
        epsilon = 1e-10
        rolles_condition = abs(float(f_a - f_b)) < epsilon
        if not rolles_condition:
            return [], False
        f_prime = sp.diff(f, x)
        print(f"Derivative for Rolle's Theorem: f'(x) = {f_prime}")
        class SolveWithTimeout:
            def __init__(self):
                self.solutions = []
                self.exception = None
                self.completed = False
            def solve_func(self):
                try:
                    self.solutions = sp.solve(f_prime, x)
                    self.completed = True
                except Exception as e:
                    self.exception = e
                    self.completed = True
        solver = SolveWithTimeout()
        thread = threading.Thread(target=solver.solve_func)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if solver.completed and solver.exception is None:
            c_values = [float(sol) for sol in solver.solutions if sol.is_real and a < float(sol) < b]
            if c_values:
                return c_values, rolles_condition
            else:
                print("No symbolic solutions found; trying numerical methods")
        else:
            print("Symbolic solution timed out or failed; switching to numerical methods")
        print("Attempting numerical solution for Rolle's Theorem...")
        c_values = multi_method_solver(f_prime, 0, a, b)
        if c_values:
            return c_values, rolles_condition
        else:
            print("Attempting thorough numerical search for Rolle's Theorem...")
            c_values = adaptive_numerical_solver(f_prime, 0, a, b, max_points=50, tol=1e-10)
            return c_values, rolles_condition
    except Exception as e:
        print(f"Error in enhanced_compute_rmvt: {e}")
    return [], rolles_condition

def check_domain_validity(f, a, b, num_points=20):
    """
    Check if a function f is valid over [a, b] by sampling points.
    Returns (is_valid, invalid_points, warnings).
    """
    test_points = np.linspace(a, b, num_points)
    invalid_points = []
    warnings_list = []
    expr_str = str(f)
    fractional_power_pattern = r'\*\*\s*\(\s*1\s*/\s*(\d+)\s*\)'
    root_matches = re.findall(fractional_power_pattern, expr_str)
    if root_matches:
        for n in root_matches:
            if int(n) % 2 == 0:
                warnings_list.append(f"Function contains {n}th roots which may produce complex values for negative inputs.")
            else:
                warnings_list.append(f"Function contains {n}th roots which may need special handling for negative inputs.")
    elif 'sqrt' in expr_str:
        warnings_list.append("Function contains square roots which may produce complex values for negative inputs.")
    if '/' in str(f):
        warnings_list.append("Function contains division. Check for removable discontinuities.")
    def safe_eval(expr, point):
        try:
            result = expr.subs(x, point)
            def process_pow_expression(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q > 1:
                        return sp.sign(base)**(1/exp.q) * (sp.Abs(base) ** exp)
                if expr.args:
                    new_args = [process_pow_expression(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            if 'I' in str(result):
                result = process_pow_expression(expr).subs(x, point)
            float_result = float(result)
            return float_result, np.isnan(float_result) or np.isinf(float_result)
        except Exception as e:
            return None, True
    for point in test_points:
        result, is_invalid = safe_eval(f, point)
        if is_invalid:
            invalid_points.append(point)
            if result is None:
                try:
                    f.subs(x, point)
                except Exception as e:
                    err_msg = str(e)
                    if "cannot convert complex" in err_msg:
                        warnings_list.append(f"Complex values detected near x = {point}")
                    elif "division by zero" in err_msg:
                        warnings_list.append(f"Division by zero detected near x = {point}")
    return len(invalid_points) == 0, invalid_points, warnings_list

def enhanced_plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope):
    """
    Enhanced visualization of f, secant line, and LMVT/RMVT points.
    """
    f_num = sp.lambdify(x, f, 'numpy')
    padding = (b - a) * 0.2
    x_min, x_max = a - padding, b + padding
    x_vals = np.linspace(x_min, x_max, 1000)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(valid_x) > 3:
        try:
            spl = make_interp_spline(valid_x, y_vals, k=3)
            smooth_x = np.linspace(min(valid_x), max(valid_x), 1000)
            smooth_y = spl(smooth_x)
            ax.plot(smooth_x, smooth_y, 'b-', linewidth=2, label=f'f(x) = {f}')
        except Exception:
            ax.plot(valid_x, y_vals, 'b-', linewidth=2, label=f'f(x) = {f}')
    else:
        ax.plot(valid_x, y_vals, 'b-', linewidth=2, label=f'f(x) = {f}')
    f_a = float(f.subs(x, a))
    f_b = float(f.subs(x, b))
    ax.plot([a, b], [f_a, f_b], 'r--', linewidth=1.5, label=f'Secant line (slope = {avg_slope:.4f})')
    ax.axvline(x=a, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=b, color='gray', linestyle=':', alpha=0.7)
    ax.plot(a, f_a, 'ro', markersize=6)
    ax.plot(b, f_b, 'ro', markersize=6)
    ax.annotate(f'({a}, {f_a:.4f})', (a, f_a), xytext=(-10, -20), textcoords='offset points', ha='right')
    ax.annotate(f'({b}, {f_b:.4f})', (b, f_b), xytext=(10, -20), textcoords='offset points', ha='left')
    try:
        x_interval = np.linspace(a, b, 100)
        y_interval = [float(f.subs(x, xi)) for xi in x_interval]
        verts = [(a, 0)] + list(zip(x_interval, y_interval)) + [(b, 0)]
        poly = Polygon(verts, facecolor='lightblue', edgecolor='none', alpha=0.3)
        ax.add_patch(poly)
    except Exception:
        pass
    for c in lmvt_points:
        f_c = float(f.subs(x, c))
        f_prime = sp.diff(f, x)
        slope = float(f_prime.subs(x, c))
        tangent_x = np.linspace(c - (b-a)/4, c + (b-a)/4, 2)
        tangent_y = [f_c + slope * (xx - c) for xx in tangent_x]
        ax.plot(tangent_x, tangent_y, 'g-', linewidth=1.5)
        ax.plot(c, f_c, 'go', markersize=8)
        ax.annotate(f'LMVT: ({c:.4f}, {f_c:.4f})', (c, f_c), xytext=(0, 20), textcoords='offset points', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    ax.set_title("Mean Value Theorem Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    plt.tight_layout()
    plt.show()

#######################################
#         Main Program Code           #
#######################################

# Custom log function to support two-argument log calls.
def custom_log(*args):
    if len(args) == 1:
        return sp.log(args[0])
    elif len(args) == 2:
        base, value = args
        return sp.log(value, base)
    else:
        raise ValueError("log function takes one or two arguments.")

def get_function_input():
    """
    Get a function input from the user and convert it to a SymPy expression.
    """
    while True:
        try:
            transformations = sp.parsing.sympy_parser.standard_transformations + (sp.parsing.sympy_parser.implicit_multiplication_application,)
            local_dict = {
                'log': custom_log,
                'pi': sp.pi,
                'e': sp.E,
                'exp': sp.exp,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'sec': lambda x: 1/sp.cos(x),
                'csc': lambda x: 1/sp.sin(x),
                'cot': lambda x: 1/sp.tan(x),
                'arcsin': sp.asin,
                'arccos': sp.acos,
                'arctan': sp.atan,
                'asin': sp.asin,
                'acos': sp.acos,
                'atan': sp.atan,
                'arcsec': lambda x: sp.acos(1/x),
                'arccsc': lambda x: sp.asin(1/x),
                'arccot': lambda x: sp.atan(1/x),
                'asec': lambda x: sp.acos(1/x),
                'acsc': lambda x: sp.asin(1/x),
                'acot': lambda x: sp.atan(1/x),
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                'cbrt': lambda x: sp.sign(x) * (sp.Abs(x) ** (sp.Rational(1, 3))),
                'nthroot': lambda x, n: sp.sign(x)**(1/n) * (sp.Abs(x) ** (sp.Rational(1, n)))
            }
            print("\nFunction input examples:")
            print("  Polynomials: x**2 - 2*x + 1")
            print("  Exponential: exp(x) or exp(2*x)")
            print("  Logarithmic: log(x) for natural log, log(10,x) for log base 10")
            print("  Trigonometric: sin(x), cos(x), tan(x)")
            print("  Inverse Trigonometric: arcsin(x), acos(x), arctan(x)")
            print("  Combined: x**2 * exp(-x) or log(4,x)/x")
            print("  Roots: cbrt(x) for cube root, sqrt(x) for square root, x**(1/n) for nth root")
            func_str = input("\nEnter a function f(x): ").strip()
            func_str = re.sub(r'(\d*\.?\d+)[eE]\^', r'\1*exp', func_str)
            func_str = re.sub(r'(arc|a)(sin|cos|tan|sec|csc|cot)', r'__\1\2__', func_str)
            func_str = re.sub(r'\s+', '', func_str)
            pattern = r'x\*\*\s*\(\s*1\s*/\s*(\d+)\s*\)'
            matches = re.findall(pattern, func_str)
            for n in matches:
                func_str = re.sub(r'x\*\*\s*\(\s*1\s*/\s*' + n + r'\s*\)', f'nthroot(x,{n})', func_str)
            func_str = re.sub(r'x\*\*\(1\.0/3\.0\)', r'cbrt(x)', func_str)
            func_str = re.sub(r'x\*\*\(1\.0/3\)', r'cbrt(x)', func_str)
            func_str = re.sub(r'x\*\*\(1/3\.0\)', r'cbrt(x)', func_str)
            func_str = re.sub(r'e\^([x\(\w])', r'exp(\1)', func_str)
            func_str = re.sub(r'e\*\*', 'exp', func_str)
            func_str = re.sub(r'\*\*\s*', '**', func_str)
            func_str = re.sub(r'ln\(', 'log(', func_str)
            func_str = re.sub(r'sec\((.*?)\)', r'1/cos(\1)', func_str)
            func_str = re.sub(r'csc\((.*?)\)', r'1/sin(\1)', func_str)
            func_str = re.sub(r'cot\((.*?)\)', r'1/tan(\1)', func_str)
            func_str = re.sub(r'sin\^-1\((.*?)\)', r'arcsin(\1)', func_str)
            func_str = re.sub(r'cos\^-1\((.*?)\)', r'arccos(\1)', func_str)
            func_str = re.sub(r'tan\^-1\((.*?)\)', r'arctan(\1)', func_str)
            func_str = re.sub(r'sec\^-1\((.*?)\)', r'arcsec(\1)', func_str)
            func_str = re.sub(r'csc\^-1\((.*?)\)', r'arccsc(\1)', func_str)
            func_str = re.sub(r'cot\^-1\((.*?)\)', r'arccot(\1)', func_str)
            func_str = re.sub(r'__arc(sin|cos|tan|sec|csc|cot)__', r'arc\1', func_str)
            func_str = re.sub(r'__a(sin|cos|tan|sec|csc|cot)__', r'a\1', func_str)
            func_str = re.sub(r'([+\-*/])(?!\*)', r' \1 ', func_str)
            func_str = re.sub(r'\*\*', '**', func_str)
            print(f"Processing function: {func_str}")
            f = sp.parsing.sympy_parser.parse_expr(func_str, transformations=transformations, local_dict=local_dict)
            def process_powers(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q > 1:
                        n = exp.q
                        if n % 2 == 0:
                            return sp.sign(base)**(1/n) * (sp.Abs(base) ** exp)
                        else:
                            return sp.sign(base) * (sp.Abs(base) ** exp)
                if expr.args:
                    new_args = [process_powers(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            f = process_powers(f)
            f = sp.simplify(f)
            test_points = [-2, -1, 1, 2]
            valid_points = False
            for point in test_points:
                try:
                    test_val = float(f.subs(x, point))
                    if not (np.isnan(test_val) or np.isinf(test_val)):
                        valid_points = True
                        break
                except Exception as e:
                    continue
            if valid_points:
                return f
            raise ValueError("Function could not be evaluated at any test points")
        except Exception as e:
            print(f"\nError parsing function: {str(e)}")
            print("\nPlease try again with a valid mathematical expression.")
            print("Tips:")
            print("1. For exponential functions, use 'exp(x)' or 'e^x'")
            print("2. For cube root, use 'cbrt(x)' or x**(1/3)")
            print("3. For nth root, use 'nthroot(x,n)' or x**(1/n)")
            print("4. For square root, use 'sqrt(x)' or x**(1/2)")
            print("5. For logarithms, use 'log(x)' or 'log(base,x)'")
            print("6. For trigonometric functions, ensure arguments are properly parenthesized")
            print("7. Ensure all variables are 'x' and all functions are supported\n")
        # Loop until a valid function is entered.
        
def get_interval_input():
    """
    Get the interval [a, b] from the user.
    """
    while True:
        try:
            a_str = input("Enter the left endpoint of the interval (a): ").strip()
            b_str = input("Enter the right endpoint of the interval (b): ").strip()
            a = float(sp.sympify(a_str, evaluate=True).evalf())
            b = float(sp.sympify(b_str, evaluate=True).evalf())
            if a >= b:
                print("Error: a must be less than b.")
                continue
            return a, b
        except Exception as e:
            print("Error: Please enter valid numerical values for the interval endpoints.")
            print("You can use constants like 'pi' or 'e'.")

def display_results(f, a, b, lmvt_points, avg_slope, rmvt_points, rolles_condition):
    """
    Display LMVT and Rolle's Theorem results.
    """
    f_prime = sp.diff(f, x)
    print("\n" + "=" * 60)
    print(f"Function: f(x) = {f}")
    print(f"Interval: [{a}, {b}]")
    print(f"f({a}) = {float(f.subs(x, a)):.6f}")
    print(f"f({b}) = {float(f.subs(x, b)):.6f}")
    print(f"f'(x) = {f_prime}")
    print("=" * 60)
    print("\nLagrange Mean Value Theorem Results:")
    print(f"Average slope over [{a}, {b}]: {avg_slope:.6f}")
    if lmvt_points:
        print(f"Found {len(lmvt_points)} point(s) where f'(c) equals the average slope:")
        for i, c in enumerate(lmvt_points, 1):
            f_c = float(f.subs(x, c))
            f_prime_c = float(f_prime.subs(x, c))
            print(f"  Point {i}: c = {c:.6f}, f(c) = {f_c:.6f}, f'(c) = {f_prime_c:.6f}")
    else:
        print("No points found that satisfy the Lagrange Mean Value Theorem.")
    print("\nRolle's Theorem Results:")
    if rolles_condition:
        print(f"Rolle's Theorem is applicable because f({a}) ≈ f({b}).")
        if rmvt_points:
            print(f"Found {len(rmvt_points)} point(s) where f'(c) = 0:")
            for i, c in enumerate(rmvt_points, 1):
                f_c = float(f.subs(x, c))
                print(f"  Point {i}: c = {c:.6f}, f(c) = {f_c:.6f}, f'(c) = 0")
        else:
            print("No points found that satisfy Rolle's Theorem.")
    else:
        print(f"Rolle's Theorem is not applicable because f({a}) ≠ f({b}).")

def main():
    """
    Main function to run the Mean Value Theorem Calculator.
    """
    print("Welcome to the Mean Value Theorem Calculator!")
    print("This program calculates points guaranteed by the Lagrange Mean Value Theorem")
    print("and Rolle's Theorem for a given function and interval.\n")
    # Determine if enhanced functions are available (we merged them here)
    use_enhanced = True
    while True:
        f = get_function_input()
        a, b = get_interval_input()
        if use_enhanced:
            is_valid, invalid_points, domain_warnings = check_domain_validity(f, a, b)
            if domain_warnings:
                print("\nDomain Warnings:")
                for warning in domain_warnings:
                    print(f"- {warning}")
            if not is_valid:
                print("\nWarning: The function may not be valid at all points in the interval.")
            lmvt_points, avg_slope = enhanced_compute_lmvt(f, a, b)
            rmvt_points, rolles_condition = enhanced_compute_rmvt(f, a, b)
        else:
            lmvt_points, avg_slope = compute_lmvt(f, a, b)
            rmvt_points, rolles_condition = compute_rmvt(f, a, b)
        display_results(f, a, b, lmvt_points, avg_slope, rmvt_points, rolles_condition)
        try:
            if use_enhanced:
                enhanced_plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope)
            else:
                plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope)
        except Exception as e:
            print(f"Error during plotting: {e}")
        cont = input("\nWould you like to try another function? (y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()
