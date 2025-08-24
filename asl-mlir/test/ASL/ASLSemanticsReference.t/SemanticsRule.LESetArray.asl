// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  var my_array: array [[42]] of integer;
  my_array[[3]] = 53;
  assert my_array[[3]] == 53;

  return 0;
end;
