#!/usr/bin/env python3
"""
Differential Testing Script for GMP Dialect

This script compares the constant folding results of the GMP dialect with
the actual GMP library results to ensure correctness. It generates test
cases, runs them through the compiler, and validates the output.

Requirements:
- Python 3.7+
- gmpy2 (for GMP library access)
- asl-opt built

Usage:
    python3 differential-test.py [--asl-opt PATH] [--verbose] [--num-tests N]
"""

import argparse
import subprocess
import tempfile
import os
import sys
import random
from fractions import Fraction

try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False
    print("Warning: gmpy2 not found. Using Python's arbitrary precision integers.")

def generate_random_int(bits=64):
    """Generate a random arbitrary precision integer."""
    value = random.randint(-(2**(bits-1)), 2**(bits-1) - 1)
    return value

def generate_random_positive_int(bits=32):
    """Generate a random positive integer."""
    return random.randint(1, 2**bits - 1)

def generate_random_fraction():
    """Generate a random fraction."""
    num = generate_random_int(32)
    den = generate_random_positive_int(16)
    return Fraction(num, den)

class TestCase:
    """Base class for test cases."""
    def __init__(self, name, mlir_code, expected_result):
        self.name = name
        self.mlir_code = mlir_code
        self.expected_result = expected_result

def z_add_test(a, b):
    expected = a + b
    mlir = f'''
func.func @test_z_add() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}}
'''
    return TestCase(f"z_add({a}, {b})", mlir, str(expected))

def z_sub_test(a, b):
    expected = a - b
    mlir = f'''
func.func @test_z_sub() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}}
'''
    return TestCase(f"z_sub({a}, {b})", mlir, str(expected))

def z_mul_test(a, b):
    expected = a * b
    mlir = f'''
func.func @test_z_mul() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}}
'''
    return TestCase(f"z_mul({a}, {b})", mlir, str(expected))

def z_div_test(a, b):
    """Truncated division toward zero."""
    if b == 0:
        return None
    # Python's // is floor division, we need truncated division
    expected = int(a / b)  # truncate toward zero
    mlir = f'''
func.func @test_z_div() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %quot = gmp.z.div %a, %b
  return %quot : !gmp.z
}}
'''
    return TestCase(f"z_div({a}, {b})", mlir, str(expected))

def z_neg_test(a):
    expected = -a
    mlir = f'''
func.func @test_z_neg() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %neg = gmp.z.neg %a
  return %neg : !gmp.z
}}
'''
    return TestCase(f"z_neg({a})", mlir, str(expected))

def z_abs_test(a):
    expected = abs(a)
    mlir = f'''
func.func @test_z_abs() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %abs = gmp.z.abs %a
  return %abs : !gmp.z
}}
'''
    return TestCase(f"z_abs({a})", mlir, str(expected))

def z_compare_test(a, b):
    if a < b:
        expected = -1
    elif a > b:
        expected = 1
    else:
        expected = 0
    mlir = f'''
func.func @test_z_compare() -> i32 {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}}
'''
    return TestCase(f"z_compare({a}, {b})", mlir, str(expected))

def z_equal_test(a, b):
    expected = "true" if a == b else "false"
    mlir = f'''
func.func @test_z_equal() -> i1 {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}}
'''
    return TestCase(f"z_equal({a}, {b})", mlir, expected)

def z_gcd_test(a, b):
    from math import gcd
    expected = gcd(abs(a), abs(b))
    mlir = f'''
func.func @test_z_gcd() -> !gmp.z {{
  %a = gmp.z.constant #gmp.z<{a}>
  %b = gmp.z.constant #gmp.z<{b}>
  %g = gmp.z.gcd %a, %b
  return %g : !gmp.z
}}
'''
    return TestCase(f"z_gcd({a}, {b})", mlir, str(expected))

def z_pow_test(base, exp):
    if exp < 0:
        return None
    if exp > 1000:  # Limit exponent size to avoid huge numbers
        exp = exp % 1001
    expected = base ** exp
    mlir = f'''
func.func @test_z_pow() -> !gmp.z {{
  %base = gmp.z.constant #gmp.z<{base}>
  %exp = arith.constant {exp} : i64
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}}
'''
    return TestCase(f"z_pow({base}, {exp})", mlir, str(expected))

def q_add_test(a, b):
    """Test rational addition."""
    expected = a + b
    mlir = f'''
func.func @test_q_add() -> !gmp.q {{
  %a = gmp.q.constant #gmp.q<{a.numerator}, {a.denominator}>
  %b = gmp.q.constant #gmp.q<{b.numerator}, {b.denominator}>
  %sum = gmp.q.add %a, %b
  return %sum : !gmp.q
}}
'''
    return TestCase(f"q_add({a}, {b})", mlir, f"{expected.numerator}/{expected.denominator}")

def q_mul_test(a, b):
    """Test rational multiplication."""
    expected = a * b
    mlir = f'''
func.func @test_q_mul() -> !gmp.q {{
  %a = gmp.q.constant #gmp.q<{a.numerator}, {a.denominator}>
  %b = gmp.q.constant #gmp.q<{b.numerator}, {b.denominator}>
  %prod = gmp.q.mul %a, %b
  return %prod : !gmp.q
}}
'''
    return TestCase(f"q_mul({a}, {b})", mlir, f"{expected.numerator}/{expected.denominator}")

def q_neg_test(a):
    """Test rational negation."""
    expected = -a
    mlir = f'''
func.func @test_q_neg() -> !gmp.q {{
  %a = gmp.q.constant #gmp.q<{a.numerator}, {a.denominator}>
  %neg = gmp.q.neg %a
  return %neg : !gmp.q
}}
'''
    return TestCase(f"q_neg({a})", mlir, f"{expected.numerator}/{expected.denominator}")

def q_equal_test(a, b):
    """Test rational equality."""
    expected = "true" if a == b else "false"
    mlir = f'''
func.func @test_q_equal() -> i1 {{
  %a = gmp.q.constant #gmp.q<{a.numerator}, {a.denominator}>
  %b = gmp.q.constant #gmp.q<{b.numerator}, {b.denominator}>
  %eq = gmp.q.equal %a, %b
  return %eq : i1
}}
'''
    return TestCase(f"q_equal({a}, {b})", mlir, expected)

def q_floor_test(a):
    """Test floor of rational."""
    from math import floor
    expected = floor(a)
    mlir = f'''
func.func @test_q_floor() -> !gmp.z {{
  %a = gmp.q.constant #gmp.q<{a.numerator}, {a.denominator}>
  %floor = gmp.q.floor %a
  return %floor : !gmp.z
}}
'''
    return TestCase(f"q_floor({a})", mlir, str(expected))

def run_asl_opt(asl_opt_path, mlir_code):
    """Run asl-opt with canonicalization and extract the result."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_code)
        f.flush()
        temp_path = f.name

    try:
        result = subprocess.run(
            [asl_opt_path, temp_path, '--canonicalize'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(temp_path)

def extract_result(output, test_name):
    """Extract the folded constant result from the output."""
    # Look for patterns like:
    # %0 = gmp.z.constant <123>
    # %true = arith.constant true
    # %c-1_i32 = arith.constant -1 : i32

    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()

        # Z/Q constant pattern
        if 'gmp.z.constant' in line or 'gmp.q.constant' in line:
            if '<' in line and '>' in line:
                start = line.index('<') + 1
                end = line.rindex('>')
                return line[start:end]

        # arith.constant for booleans
        if 'arith.constant true' in line:
            return 'true'
        if 'arith.constant false' in line:
            return 'false'

        # arith.constant for integers (comparison results)
        if 'arith.constant' in line and 'i32' in line:
            # Pattern: %c-1_i32 = arith.constant -1 : i32
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'arith.constant' and i + 1 < len(parts):
                    value = parts[i + 1]
                    if ':' in value:
                        value = value.split(':')[0].strip()
                    return value

    return None

def run_test(asl_opt_path, test_case, verbose=False):
    """Run a single test case and check the result."""
    stdout, stderr, returncode = run_asl_opt(asl_opt_path, test_case.mlir_code)

    if returncode != 0:
        if verbose:
            print(f"FAIL: {test_case.name} - asl-opt failed")
            print(f"  stderr: {stderr[:200]}")
        return False, "asl-opt failed"

    result = extract_result(stdout, test_case.name)

    if result is None:
        if verbose:
            print(f"FAIL: {test_case.name} - could not extract result")
            print(f"  output: {stdout[:200]}")
        return False, "could not extract result"

    # Normalize comparison
    expected = test_case.expected_result.strip()
    result = result.strip()

    # Handle rational comparison (n/d format)
    if '/' in expected and ',' in result:
        # Convert "num, den" to "num/den"
        result = result.replace(', ', '/')

    if result == expected:
        if verbose:
            print(f"PASS: {test_case.name} = {result}")
        return True, result
    else:
        if verbose:
            print(f"FAIL: {test_case.name}")
            print(f"  expected: {expected}")
            print(f"  got:      {result}")
        return False, result

def main():
    parser = argparse.ArgumentParser(description='Differential testing for GMP dialect')
    parser.add_argument('--asl-opt', default='./build/bin/asl-opt',
                       help='Path to asl-opt binary')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    parser.add_argument('--num-tests', '-n', type=int, default=100,
                       help='Number of random tests per operation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    # Check asl-opt exists
    if not os.path.exists(args.asl_opt):
        print(f"Error: asl-opt not found at {args.asl_opt}")
        sys.exit(1)

    tests = []

    # Generate Z integer tests
    print(f"Generating {args.num_tests} tests per operation...")

    for _ in range(args.num_tests):
        a = generate_random_int(64)
        b = generate_random_int(64)

        tests.append(z_add_test(a, b))
        tests.append(z_sub_test(a, b))
        tests.append(z_mul_test(a, b))
        tests.append(z_neg_test(a))
        tests.append(z_abs_test(a))
        tests.append(z_compare_test(a, b))
        tests.append(z_equal_test(a, b))
        tests.append(z_gcd_test(abs(a) + 1, abs(b) + 1))

        # Division needs non-zero divisor
        if b != 0:
            test = z_div_test(a, b)
            if test:
                tests.append(test)

        # Power with small exponents
        base = generate_random_int(8)
        exp = random.randint(0, 10)
        test = z_pow_test(base, exp)
        if test:
            tests.append(test)

    # Generate Q rational tests
    for _ in range(args.num_tests):
        a = generate_random_fraction()
        b = generate_random_fraction()

        tests.append(q_add_test(a, b))
        tests.append(q_mul_test(a, b))
        tests.append(q_neg_test(a))
        tests.append(q_equal_test(a, b))
        tests.append(q_equal_test(a, a))  # Should always be true
        tests.append(q_floor_test(a))

    # Run all tests
    passed = 0
    failed = 0

    print(f"\nRunning {len(tests)} tests...")

    for test in tests:
        if test is None:
            continue
        success, _ = run_test(args.asl_opt, test, args.verbose)
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
