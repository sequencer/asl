// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func compare_integers{
    int1: integer {1,2},
    int2: integer {4,5},
    int3: integer}(a: bits(int1), b: bits(int2), c: bits(int3))
begin
    var cond: boolean;
    cond = int1 == int2; // Legal
    cond = int1 == int3; // Legal
end;
