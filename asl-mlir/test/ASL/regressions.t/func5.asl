// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({


readonly func f(i : integer) => integer
begin
  return i + 2;
end;

readonly func f(b : boolean) => boolean
begin
  return if b then FALSE else TRUE;
end;

readonly func f(x : bits(3)) => boolean
begin
  return x[0] == '0';
end;

func main() => integer
begin
  assert f(0) == 2;
  assert f(1) == 3;
  assert f(FALSE);
  assert f('110');

  return 0;
end;


