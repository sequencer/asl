// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func increment(x: integer) => integer
begin
    return x + 1;
end;

func main() => integer
begin
    - = 42;
    - = increment(42);
    var (-, x) = (42, 43);
    return 0;
end;
