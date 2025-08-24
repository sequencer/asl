// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// assignment is legal as both are the same under constrained integer
func positive5{N}(bv : bits(N))
begin
    var a = 0 as integer {0..N};
    var b = 0 as integer {0..N};

    a = b;
end;
