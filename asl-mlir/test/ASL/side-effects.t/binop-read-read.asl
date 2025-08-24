// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var X: integer = 0;

func main () => integer
begin
  let x = X + X;

  return 0;
end;
