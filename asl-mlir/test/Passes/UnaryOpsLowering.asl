// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

// Test Phase 3: Unary Operations lowering to EmitC

// Boolean NOT
// CHECK: !(
func BoolNot(a: boolean) => boolean
begin
  return !a;
end

// Integer negation
// CHECK: mpz_neg
func IntNeg(a: integer) => integer
begin
  return -a;
end

// Real negation
// CHECK: mpq_neg
func RealNeg(a: real) => real
begin
  return -a;
end

// Bitvector NOT
// CHECK: ~(
func BitsNot(a: bits(8)) => bits(8)
begin
  return NOT a;
end
