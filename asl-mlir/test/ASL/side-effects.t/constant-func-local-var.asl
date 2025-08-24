// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func foo (x: integer) => integer
begin
  var result: integer = 3;
  result = result + x * x;
  return result;
end;

constant C = foo (4);

func main () => integer
begin
  assert C == 19;

  return 0;
end;

