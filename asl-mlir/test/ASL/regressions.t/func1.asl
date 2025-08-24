// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func f(i:integer) => integer
begin
    return i;
end;

func main() => integer
begin
    let x = 3;
    let y = f(x);
    assert x == 3;
    assert y == 3;

    return 0;
end;


