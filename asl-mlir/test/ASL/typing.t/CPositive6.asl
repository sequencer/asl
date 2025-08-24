// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// parameter N can be assigned to R of unconstrained integer type
func positive6{N}(x: bits(N))
begin
    var R : integer = N;
end;
