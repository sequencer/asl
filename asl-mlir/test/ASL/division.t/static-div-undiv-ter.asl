// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let a = ARBITRARY : integer {2, 3};
  let b = a DIV 2 as integer {1};

  return 0;
end;

