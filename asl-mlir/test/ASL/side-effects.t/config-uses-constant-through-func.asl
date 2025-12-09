// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

constant X: integer = 0;

pure func foo () => integer
begin
  return X;
end;

config Y: integer = foo ();

func main () => integer
begin
  assert (Y == 0);

  return 0;
end;
