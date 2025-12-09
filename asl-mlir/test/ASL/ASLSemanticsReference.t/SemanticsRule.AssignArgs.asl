// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func plus(x: integer, y: integer) => integer
begin
    return x + y;
end;

func main() => integer
begin
    - = plus(10, 5);
    return 0;
end;
