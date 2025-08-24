// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func divide(a: integer, b: integer) => integer
begin
    return a DIV b;
end;

func main() => integer
begin
    var x = divide(128, 7);
    return 0;
end;
