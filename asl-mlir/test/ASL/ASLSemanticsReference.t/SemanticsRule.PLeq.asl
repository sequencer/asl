// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let match_true = 3 IN { <= 42 };
  assert match_true == TRUE;

  let match_false = 42 IN { <= 3 };
  assert match_false == FALSE;

  return 0;
end;
