// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({



func f() => (integer, integer, integer)
begin
  return (3, 4, 5);
end;

func multiple_return_values ()
begin
  let (a, b, c) = f();
  assert a == 3;
  assert b == 4;
  assert c == 5;
end;

func other_tuple_usages ()
begin
  let t = f();
  let (a, b, c) = t;
  assert a == 3;
  assert b == 4;
  assert c == 5;
end;

func with_var ()
begin
  var a, b, c: integer;
  (a, b, c) = f();
  assert a == 3;
  assert b == 4;
  assert c == 5;
end;

func main() => integer
begin
  multiple_return_values();
  other_tuple_usages();
  with_var ();

  return 0;
end;


