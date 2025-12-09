// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

constant X: integer {0..100} = 0;

type T of integer {X};

func main () => integer
begin
  return 0;
end;

