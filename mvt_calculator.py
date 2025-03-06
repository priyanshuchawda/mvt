#!/usr/bin/env python3
"""
Mean Value Theorem Calculator - Main Program

This program provides a command-line interface for users to calculate the points
guaranteed by the Lagrange Mean Value Theorem and Rolle's Theorem for a given function
and interval.

Usage:
    python mvt_calculator.py
"""

import sympy as sp
import warnings
import mpmath as mp
from sympy import Symbol, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Import basic functions from mean_value_theorem module
from mean_value_theorem import compute_lmvt, compute_rmvt, plot_function_and_theorems

# Import enhanced functions from mvt_advanced module
try:
    from mvt_advanced import (
        enhanced_compute_lmvt, enhanced_compute_rmvt, 
        enhanced_plot_function_and_theorems, check_domain_validity
    )
    USE_ENHANCED_FUNCTIONS = True
    print("Using enhanced MVT functions for better handling of complex functions.")
except ImportError:
    USE_ENHANCED_FUNCTIONS = False
    print("Enhanced functions not available. Using standard functions.")

# Suppress numpy warnings for invalid operations (like log of negative numbers)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Define a custom log function
def custom_log(*args):
    """
    Custom log function that allows log(4, x) to be interpreted as log x base 4.
    If one argument is given, returns the natural log.
    If two arguments are given, returns log of the second argument with the first as base.
    """
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
    
    Returns:
        sympy.Expr: The symbolic representation of the function
    """
    while True:
        try:
            # Set up transformations for more user-friendly input
            transformations = standard_transformations + (implicit_multiplication_application,)
            
            # Set up a local dictionary to include our custom log and common constants
            local_dict = {
                'log': custom_log,  # use custom log
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
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'sqrt': sp.sqrt,
                'abs': sp.Abs
            }
            
            # Get the function from the user with examples including transcendental functions
            print("\nFunction input examples:")
            print("  Polynomials: x**2 - 2*x + 1")
            print("  Exponential: exp(x) or exp(2*x)")
            print("  Logarithmic: log(x) for natural log, log(10,x) for log base 10")
            print("  Trigonometric: sin(x), cos(x), tan(x)")
            print("  Combined: x**2 * exp(-x) or log(4,x)/x")
            print("  Complex: sin(x) + log(2,x) or exp(2*x) + sin(x)")
            func_str = input("\nEnter a function f(x): ")
            
            # Handle common user input patterns more comprehensively
            # Handle exponential expressions
            func_str = func_str.replace("e^x", "exp(x)")
            func_str = func_str.replace("e^(", "exp(")
            func_str = func_str.replace("e^", "exp")
            func_str = func_str.replace("e**", "exp")
            
            # Handle logarithmic expressions
            func_str = func_str.replace("ln(", "log(")
            
            # Ensure proper handling of nested expressions
            # Replace any remaining e^something patterns with a more comprehensive regex
            import re
            func_str = re.sub(r'e\^([a-zA-Z0-9_\+\-\*\/\(\)\s]+)', r'exp(\1)', func_str)
            
            # Parse the function string into a SymPy expression with our custom local dictionary
            x = Symbol('x')
            f = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
            
            # Test if the function can be evaluated at a valid point
            # For logarithmic functions, test at x=2 instead of x=1
            # For more complex functions, try multiple test points
            test_points = []
            if "log" in func_str.lower():
                # For logarithmic functions, ensure test point is positive
                test_points = [2, 3]
            else:
                test_points = [1, 2]
                
            # Try each test point until one works
            for point in test_points:
                try:
                    test_val = float(f.subs(x, point))
                    # If we get here without exception, the function is valid at this point
                    return f
                except Exception as e:
                    # If this test point fails, try the next one
                    if point == test_points[-1]:
                        # If this was the last test point, re-raise the exception
                        raise e
                    continue
            
            return f
        except Exception as e:
            print(f"Error parsing function: {e}")
            print("Please try again with a valid mathematical expression.")
            print("For exponential functions, use 'exp(x)' instead of 'e^x' or 'e**x'.")
            print("For logarithms, use 'log(x)' for natural logarithm or 'log(10,x)' for base 10.")
            print("Make sure your function is defined for the test points (x=1, x=2, etc.).")
            print("For complex expressions, ensure proper parentheses and syntax.")
            print("For mixed functions (like sin(x) + log(2,x)), the calculator will use numerical methods.")

def get_interval_input():
    """
    Get the interval [a, b] from the user.
    
    Returns:
        tuple: (a, b) representing the interval
    """
    while True:
        try:
            a_str = input("Enter the left endpoint of the interval (a): ").strip()
            b_str = input("Enter the right endpoint of the interval (b): ").strip()

            # Use sympy.sympify to evaluate the input expressions, so users can enter things like 'pi/16'
            a = float(sp.sympify(a_str, evaluate=True).evalf())
            b = float(sp.sympify(b_str, evaluate=True).evalf())
            
            if a >= b:
                print("Error: The left endpoint (a) must be less than the right endpoint (b).")
                continue
            
            return a, b
        except Exception as e:
            print("Error: Please enter valid numerical values for the interval endpoints.")
            print("You can use mathematical constants like 'pi' or 'e'.")

def display_results(f, a, b, lmvt_points, avg_slope, rmvt_points, rolles_condition):
    """
    Display the results of the LMVT and RMVT calculations.
    
    Args:
        f: SymPy expression representing the function
        a: Left endpoint of the interval
        b: Right endpoint of the interval
        lmvt_points: List of points satisfying LMVT
        avg_slope: The average slope over the interval
        rmvt_points: List of points satisfying RMVT
        rolles_condition: Whether f(a) = f(b) (Rolle's condition)
    """
    x = Symbol('x')
    f_prime = sp.diff(f, x)
    
    print("\n" + "=" * 60)
    print(f"Function: f(x) = {f}")
    print(f"Interval: [{a}, {b}]")
    print(f"f({a}) = {float(f.subs(x, a)):.6f}")
    print(f"f({b}) = {float(f.subs(x, b)):.6f}")
    print(f"f'(x) = {f_prime}")
    print("=" * 60)
    
    # Display LMVT results
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
        print("This may be due to numerical limitations or an error in the calculation.")
    
    # Display RMVT results
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
            print("This may be due to numerical limitations or an error in the calculation.")
    else:
        print(f"Rolle's Theorem is not applicable because f({a}) ≠ f({b}).")

def main():
    """
    Main function to run the Mean Value Theorem calculator.
    """
    print("Welcome to the Mean Value Theorem Calculator!")
    print("This program calculates points guaranteed by the Lagrange Mean Value Theorem")
    print("and Rolle's Theorem for a given function and interval.\n")
    
    while True:
        # Get user input
        f = get_function_input()
        a, b = get_interval_input()
        
        # Check domain validity if enhanced functions are available
        if USE_ENHANCED_FUNCTIONS:
            is_valid, invalid_points = check_domain_validity(f, a, b)
            if not is_valid:
                print("\nWarning: The function may not be valid at all points in the interval.")
                print(f"Points where the function is undefined or returns invalid values: {invalid_points}")
                print("The calculator will attempt to work around these points.")
        
        # Compute LMVT using enhanced functions if available
        if USE_ENHANCED_FUNCTIONS:
            lmvt_points, avg_slope = enhanced_compute_lmvt(f, a, b)
        else:
            lmvt_points, avg_slope = compute_lmvt(f, a, b)
        
        # Compute RMVT using enhanced functions if available
        if USE_ENHANCED_FUNCTIONS:
            rmvt_points, rolles_condition = enhanced_compute_rmvt(f, a, b)
        else:
            rmvt_points, rolles_condition = compute_rmvt(f, a, b)
        
        # Display results
        display_results(f, a, b, lmvt_points, avg_slope, rmvt_points, rolles_condition)
        
        # Create a plot if any points were found
        if lmvt_points or rmvt_points:
            if USE_ENHANCED_FUNCTIONS:
                enhanced_plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope)
            else:
                plot_function_and_theorems(f, a, b, lmvt_points, rmvt_points, avg_slope)
        
        # Ask if the user wants to try another function
        again = input("\nWould you like to try another function? (y/n): ").lower()
        if again != 'y':
            break
    
    print("Thank you for using the Mean Value Theorem Calculator!")

if __name__ == "__main__":
    main()