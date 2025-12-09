// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func pow_func{A, B, C, D}(
    a: integer{A},
    b: integer{B},
    c: integer{C},
    d: integer{D})
begin
    var x : integer{A..B} = ARBITRARY : integer{A..B};
    var y : integer{C..D} = ARBITRARY : integer{C..D};
    var z : integer{0..(D - 1)} = x MOD y;
end;
