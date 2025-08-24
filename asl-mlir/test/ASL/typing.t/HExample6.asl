// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func MathematicalFunction{N, M}(input : bits(N), mask : bits(M))
begin
    assert(N == M * 8);
    let p2bits = ClosestPow2(N) as integer{0..N*2};
    var op = Zeros {p2bits};
end;

func ClosestPow2(N : integer) => integer
begin
    var x = HighestSetBit(N[63:0] + 1);
    return x;
end;
