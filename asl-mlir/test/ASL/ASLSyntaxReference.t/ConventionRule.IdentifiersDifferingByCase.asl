// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func foo() => integer
begin
    return 1;
end;

func bar() => integer
begin
    return 2;
end;

func main() => integer
begin
  var color = foo();
  var Color = bar();
  // ...
  var c = color; // should this be color or Color?
  return 0;
end;
