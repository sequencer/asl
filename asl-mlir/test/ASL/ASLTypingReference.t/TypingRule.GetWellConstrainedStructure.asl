// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type counter of integer;

func foo{N}(bv: bits(N)) => integer
begin
    var x : counter = 5;
    return (-N) + (-x);
end;
