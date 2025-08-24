// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var x: integer;
  case x of
    when 2 => assert FALSE;
    when 0 where FALSE => assert FALSE;
    when 1 => assert FALSE;
    when 0 where 1 + 1 == 2 => assert TRUE;
    otherwise => assert FALSE;
  end;

  return 0;
end;
