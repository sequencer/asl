// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// N1 has the type integer{N+1}
func positive2{N}(x: bits(N)) => integer{N+1}
begin
    var N1 = N + 1;
    return N1;
end;
