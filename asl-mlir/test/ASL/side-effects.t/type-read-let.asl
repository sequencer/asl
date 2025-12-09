// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

let X: integer {8, 16} = 8;

type T of integer {X};

func main () => integer
begin
  return 0;
end;
