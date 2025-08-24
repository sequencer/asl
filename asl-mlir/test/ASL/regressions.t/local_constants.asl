// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func foo(x: integer {0..100}) => integer {0..100}
begin
    return x;
end;

func main() => integer
begin
    constant x = 32;
    constant z: integer {0..100} = foo(x);

    let bv: bits(32) = Zeros{z};
    return 0;
end;
