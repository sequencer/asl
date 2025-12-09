// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    let c1: integer{1..1000} = 42;
    let c2 = 42;

    var a: integer = 42;
    var b: integer;
    var c, d, e: integer;
    let x: integer = 42;
    let z = 42;
    return 0;
end;
