// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  case 3 of
    when 42 => assert FALSE;
    when <= 42 => assert TRUE;
    otherwise => assert FALSE;
  end;

  return 0;
end;
