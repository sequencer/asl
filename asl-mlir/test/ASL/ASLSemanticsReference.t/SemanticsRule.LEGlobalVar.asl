// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var x: integer = 3;

func main () => integer
begin

  x = 42;
  assert x==42;

  return 0;
end;
