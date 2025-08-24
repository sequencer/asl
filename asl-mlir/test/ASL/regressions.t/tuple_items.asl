// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let x = (2, 3);
  assert x.item0 == 2;
  assert x.item1 == 3;

  return 0;
end;
