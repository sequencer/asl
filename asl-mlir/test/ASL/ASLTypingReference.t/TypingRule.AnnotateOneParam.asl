// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

let ci : integer{1..1000} = 500;

func parameterized{A, B: integer, C: integer{ci}}(x: bits(A), y: bits(B), z: bits(C))
begin

    - = A;

    - = B;

    - = C;

end;



