// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
  var a = '101';
  let b = a[1, 0];
  a[0] = '0';

  assert a == '100';
  assert b == '01';

  return 0;
end;


