// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func foo () => integer
begin
  var x: integer = 0;

  return x;
end;

config Y: integer = foo ();

func main () => integer
begin
  assert (Y == 0);

  return 0;
end;
