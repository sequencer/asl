// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

// Test Phase 2: Binary Operations lowering to EmitC

// CHECK: mpz_add
func IntAdd(a: integer, b: integer) => integer
begin
  return a + b;
end

// CHECK: mpz_sub
func IntSub(a: integer, b: integer) => integer
begin
  return a - b;
end

// CHECK: mpz_mul
func IntMul(a: integer, b: integer) => integer
begin
  return a * b;
end

// CHECK: mpz_tdiv_q
func IntDiv(a: integer, b: integer) => integer
begin
  return a DIV b;
end

// CHECK: mpz_fdiv_q
func IntDivrm(a: integer, b: integer) => integer
begin
  return a DIVRM b;
end

// CHECK: mpz_fdiv_r
func IntMod(a: integer, b: integer) => integer
begin
  return a MOD b;
end

// CHECK: mpz_pow_ui
func IntPow(a: integer, b: integer) => integer
begin
  return a ^ b;
end

// CHECK: mpz_mul_2exp
func IntShl(a: integer, b: integer) => integer
begin
  return a << b;
end

// CHECK: mpz_fdiv_q_2exp
func IntShr(a: integer, b: integer) => integer
begin
  return a >> b;
end

// Bitvector operations
// CHECK: +(
func BitsAdd(a: bits(8), b: bits(8)) => bits(8)
begin
  return a + b;
end

// CHECK: -(
func BitsSub(a: bits(8), b: bits(8)) => bits(8)
begin
  return a - b;
end

// CHECK: &(
func BitsAnd(a: bits(8), b: bits(8)) => bits(8)
begin
  return a AND b;
end

// CHECK: |(
func BitsOr(a: bits(8), b: bits(8)) => bits(8)
begin
  return a OR b;
end

// CHECK: ^(
func BitsXor(a: bits(8), b: bits(8)) => bits(8)
begin
  return a XOR b;
end

// CHECK: <<(
// CHECK: |(
func BitsConcat(a: bits(4), b: bits(4)) => bits(8)
begin
  return a :: b;
end

// Real operations
// CHECK: mpq_add
func RealAdd(a: real, b: real) => real
begin
  return a + b;
end

// CHECK: mpq_sub
func RealSub(a: real, b: real) => real
begin
  return a - b;
end

// CHECK: mpq_mul
func RealMul(a: real, b: real) => real
begin
  return a * b;
end

// CHECK: mpq_div
func RealDiv(a: real, b: real) => real
begin
  return a / b;
end

// Boolean operations
// CHECK: &&(
func BoolAnd(a: boolean, b: boolean) => boolean
begin
  return a && b;
end

// CHECK: ||(
func BoolOr(a: boolean, b: boolean) => boolean
begin
  return a || b;
end

// CHECK: ==(
func BoolEq(a: boolean, b: boolean) => boolean
begin
  return a <-> b;
end

// CHECK: !(
// CHECK: ||(
func BoolImpl(a: boolean, b: boolean) => boolean
begin
  return a --> b;
end

// Comparison operations
// CHECK: mpz_cmp
// CHECK: ==(
func IntEq(a: integer, b: integer) => boolean
begin
  return a == b;
end

// CHECK: mpz_cmp
// CHECK: !=(
func IntNeq(a: integer, b: integer) => boolean
begin
  return a != b;
end

// CHECK: mpz_cmp
// CHECK: <(
func IntLt(a: integer, b: integer) => boolean
begin
  return a < b;
end

// CHECK: mpz_cmp
// CHECK: <=(
func IntLeq(a: integer, b: integer) => boolean
begin
  return a <= b;
end

// CHECK: mpz_cmp
// CHECK: >(
func IntGt(a: integer, b: integer) => boolean
begin
  return a > b;
end

// CHECK: mpz_cmp
// CHECK: >=(
func IntGeq(a: integer, b: integer) => boolean
begin
  return a >= b;
end
