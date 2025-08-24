// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let x: integer = 6;
  let y: integer = -3;
  let z = x DIVRM y;
  return 0;
end;

