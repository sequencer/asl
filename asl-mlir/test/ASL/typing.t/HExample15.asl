// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func HExample15(x: integer {4, 8}, y: integer {4, 8}, a: integer)
begin
  let d = x DIV y;
  - = a < d;
end;

