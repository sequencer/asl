// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func UNPREDICTABLE ()
begin
  assert FALSE;
end;

func main () => integer
begin
  var d: integer = ARBITRARY : integer{13, 16};
  var n: integer = d - 1;

  if d IN {13,15} || n IN {13,15} then
      UNPREDICTABLE();
  end;

  return 0;
end;
