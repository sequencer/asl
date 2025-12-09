// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
  assert Len('11') == 2;
  assert Len('110') == 3;
  assert Len('101') == 3;
  assert Len('') == 0;

  return 0;
end;


