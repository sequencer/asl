// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
  let x: integer {2, 4, 8} = 8;
  let y: integer {0, 1, 2} = 1;

  let z = x DIV y;

  return 0;
end;


