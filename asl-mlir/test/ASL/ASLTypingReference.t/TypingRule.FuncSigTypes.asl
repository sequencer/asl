// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

// The list of signature types is: integer{0..N}, real, bits(N).
func proc{N}(x: integer{0..N}, y: real, z: bits(N))
begin
    pass;
end;

// The list of signature types is: bits(N), integer{0..N}, real.
func returns_value{N}(x: integer{0..N}, y: real) => bits(N)
begin
    return Zeros{N};
end;
