// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  var x: integer = 42;
  var y: integer = 3;

  (x, y) = (3, 42);
  assert x == 3 && y == 42;

  // The above is equivalent to the following two statements:
  x = 3; y = 42;

  // Discard the element returned by f()
  (- , y) = (f(), 43);
  assert x == 3 && y == 43;
  return 0;
end;

func f() => integer begin return 5; end;
