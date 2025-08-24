// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// y has the type integer {0..N}
func positive1{N}(x: bits(N), offset: integer) => bit
begin
    var y = offset MOD N;
    return x[y];
end;
