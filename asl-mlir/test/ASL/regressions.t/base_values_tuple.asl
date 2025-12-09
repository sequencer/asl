// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  var (x, y) : (integer, boolean);

  assert x == 0 && y == FALSE;

  return 0;
end;
