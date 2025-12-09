// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

readonly func f() => integer
begin
  return 0;
end;

readonly func f(x:integer) => integer
begin
  return x;
end;

readonly func f(x:integer, y:integer) => integer
begin
  return x + y;
end;

func main() => integer
begin
  assert 0 == f();
  assert 1 == f(1);
  assert 5 == f(2, 3);

  return 0;
end;


