// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var X: integer = 0;

func main () => integer
begin
  assert X == 0;

  return 0;
end;
