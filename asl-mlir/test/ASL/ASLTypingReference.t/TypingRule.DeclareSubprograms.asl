// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

func flip_bits{N}(bv: bits(N)) => bits(N)
begin
    return Ones{N} XOR bv;
end;

func add_10(x: integer) => integer
begin
    return x + 10;
end;

func add_10(x: real) => real
begin
    return x + 10.0;
end;

func factorial(x: integer) => integer recurselimit 100
begin
    return if x == 0 then 1 else x * factorial(x - 1);
end;
