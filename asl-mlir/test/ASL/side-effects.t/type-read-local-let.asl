// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let x: integer {0..100} = 0;

  let y: integer {x} = 0;

  return 0;
end;

