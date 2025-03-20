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
import re
import numpy as np
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
                'asin': sp.asin,
                'acos': sp.acos,
                'atan': sp.atan,
                # Improved implementations for inverse trigonometric functions
                'arcsec': lambda x: sp.acos(1/x),  # arcsec(x) = arccos(1/x)
                'arccsc': lambda x: sp.asin(1/x),  # arccsc(x) = arcsin(1/x)
                'arccot': lambda x: sp.atan(1/x),  # arccot(x) = arctan(1/x)
                'asec': lambda x: sp.acos(1/x),    # Same as arcsec
                'acsc': lambda x: sp.asin(1/x),    # Same as arccsc
                'acot': lambda x: sp.atan(1/x),    # Same as arccot
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                # Add custom root functions that handle negative numbers correctly
                'cbrt': lambda x: sp.sign(x) * (sp.Abs(x) ** (sp.Rational(1, 3))),
                'nthroot': lambda x, n: sp.sign(x)**(1/n) * (sp.Abs(x) ** (sp.Rational(1, n)))
            }
            
            # Get the function from the user with examples
            print("\nFunction input examples:")
            print("  Polynomials: x**2 - 2*x + 1")
            print("  Exponential: exp(x) or exp(2*x)")
            print("  Logarithmic: log(x) for natural log, log(10,x) for log base 10")
            print("  Trigonometric: sin(x), cos(x), tan(x)")
            print("  Inverse Trigonometric: arcsin(x), asin(x), arccos(x), acos(x), arctan(x), atan(x)")
            print("  Inverse Secant/Cosecant/Cotangent: arcsec(x), asec(x), arccsc(x), acsc(x), arccot(x), acot(x)")
            print("  Combined: x**2 * exp(-x) or log(4,x)/x")
            print("  Complex: sin(x) + log(2,x) or exp(2*x) + sin(x)")
            print("  Roots: cbrt(x) for cube root, sqrt(x) for square root, x**(1/n) for nth root")
            func_str = input("\nEnter a function f(x): ").strip()
            
            # Pre-process the input string
            # First handle special cases that need preservation
            func_str = re.sub(r'(\d*\.?\d+)[eE]\^', r'\1*exp', func_str)  # Handle scientific notation
            
            # Process inverse trigonometric functions FIRST to prevent them from being broken apart
            # Replace all inverse trig functions with unique tokens that won't be split
            func_str = re.sub(r'(arc|a)(sin|cos|tan|sec|csc|cot)', r'__\1\2__', func_str)
            
            # Remove all spaces
            func_str = re.sub(r'\s+', '', func_str)
            
            # Pre-process fractional power patterns into nthroot form
            # Match x**(1/n) patterns for any n
            pattern = r'x\*\*\s*\(\s*1\s*/\s*(\d+)\s*\)'
            matches = re.findall(pattern, func_str)
            for n in matches:
                # Replace x**(1/n) with nthroot(x,n)
                func_str = re.sub(r'x\*\*\s*\(\s*1\s*/\s*' + n + r'\s*\)', f'nthroot(x,{n})', func_str)
            
            # Also handle decimal forms like x**(0.333...)
            func_str = re.sub(r'x\*\*\(1\.0/3\.0\)', r'cbrt(x)', func_str)
            func_str = re.sub(r'x\*\*\(1\.0/3\)', r'cbrt(x)', func_str)
            func_str = re.sub(r'x\*\*\(1/3\.0\)', r'cbrt(x)', func_str)
            
            # Handle exponential expressions
            func_str = re.sub(r'e\^([x\(\w])', r'exp(\1)', func_str)  # e^x or e^(...)
            func_str = re.sub(r'e\*\*', 'exp', func_str)  # e**x
            
            # Handle powers carefully - this is a key fix for x**(1/n) type expressions
            func_str = re.sub(r'\*\*\s*', '**', func_str)  # Remove spaces around **
            
            # Handle logarithmic expressions
            func_str = re.sub(r'ln\(', 'log(', func_str)
            
            # Handle trigonometric functions
            func_str = re.sub(r'sec\((.*?)\)', r'1/cos(\1)', func_str)
            func_str = re.sub(r'csc\((.*?)\)', r'1/sin(\1)', func_str)
            func_str = re.sub(r'cot\((.*?)\)', r'1/tan(\1)', func_str)
            
            # Handle inverse trigonometric functions with alternative notations
            func_str = re.sub(r'sin\^-1\((.*?)\)', r'arcsin(\1)', func_str)
            func_str = re.sub(r'cos\^-1\((.*?)\)', r'arccos(\1)', func_str)
            func_str = re.sub(r'tan\^-1\((.*?)\)', r'arctan(\1)', func_str)
            func_str = re.sub(r'sec\^-1\((.*?)\)', r'arcsec(\1)', func_str)
            func_str = re.sub(r'csc\^-1\((.*?)\)', r'arccsc(\1)', func_str)
            func_str = re.sub(r'cot\^-1\((.*?)\)', r'arccot(\1)', func_str)
            
            # Restore the tokenized inverse trig functions
            func_str = re.sub(r'__arc(sin|cos|tan|sec|csc|cot)__', r'arc\1', func_str)
            func_str = re.sub(r'__a(sin|cos|tan|sec|csc|cot)__', r'a\1', func_str)
            
            # Add minimal spaces around operators for readability in error messages
            # Modify the regex to exclude ** from being affected by the spacing
            func_str = re.sub(r'([+\-*/])(?!\*)', r' \1 ', func_str)
            
            # Handle ** separately to ensure it stays together without spaces
            func_str = re.sub(r'\*\*', '**', func_str)
            
            print(f"Processing function: {func_str}")
            
            # Parse the function string into a SymPy expression
            x = Symbol('x')
            f = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
            
            # Handle fractional powers by converting them to appropriate root expressions
            def process_powers(expr):
                if isinstance(expr, sp.Pow):
                    base, exp = expr.args
                    if isinstance(exp, sp.Rational) and exp.p == 1 and exp.q > 1:
                        # For any fractional power 1/n
                        n = exp.q
                        # For even roots (which could produce complex results with negative inputs)
                        if n % 2 == 0:
                            return sp.sign(base)**(1/n) * (sp.Abs(base) ** exp)
                        # For odd roots (which should work with negative inputs)
                        else:
                            # Use direct handling that preserves the sign
                            return sp.sign(base) * (sp.Abs(base) ** exp)
                    
                # If not a fractional power or for composite expressions, process recursively
                if expr.args:
                    new_args = [process_powers(arg) for arg in expr.args]
                    if new_args != list(expr.args):
                        return expr.func(*new_args)
                return expr
            
            # Process all fractional powers in the expression
            f = process_powers(f)
            
            # Simplify removable discontinuities if possible
            f = sp.simplify(f)
            
            # Test both positive and negative points
            test_points = [-2, -1, 1, 2]
            valid_points = False
            
            for point in test_points:
                try:
                    test_val = float(f.subs(x, point))
                    if not (np.isnan(test_val) or np.isinf(test_val)):
                        valid_points = True
                        break
                except Exception as e:
                    if point == test_points[-1]:
                        print(f"Warning: Function evaluation failed at x={point}: {str(e)}")
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
            is_valid, invalid_points, domain_warnings = check_domain_validity(f, a, b)
            if domain_warnings:
                print("\nDomain Warnings:")
                for warning in domain_warnings:
                    print(f"- {warning}")
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