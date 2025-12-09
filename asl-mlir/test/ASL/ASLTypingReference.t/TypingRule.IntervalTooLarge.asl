// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

constant A = 1 << 2;

func foo(a: integer {0..A})
begin
    var z: integer{} = a * 2;
    // The type of z is detailed in the type of w below.
    var w: integer {0, 2, 4, 6, 8} = (a * 2);
end;
