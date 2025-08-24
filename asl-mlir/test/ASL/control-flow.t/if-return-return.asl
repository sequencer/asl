// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

readonly func sign (n: integer) => integer
begin
  if n <= 0 then return -1;
  else return 1;
  end;
end;

func main () => integer
begin
  assert (sign (-1) == -1);
  assert (sign (-2) == -1);
  assert (sign (2) == 1);

  return 0;
end;
