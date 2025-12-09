// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Color of enumeration {RED, GREEN, BLUE};

func main () => integer
begin

  var my_array: array [[Color]] of integer;
  my_array[[RED]] = 53;
  assert my_array[[RED]] == 53;

  return 0;
end;
