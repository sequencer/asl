// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  
  var x,y,z : integer;

  assert x == 0 && y == 0 && z == 0;

  return 0;
end;
