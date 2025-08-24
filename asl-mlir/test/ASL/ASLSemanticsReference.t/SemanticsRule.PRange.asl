// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let match_true = 42 IN {3..42};
  assert match_true == TRUE;

  let match_false = 1 IN {3..42};
  assert match_false == FALSE;

  return 0;
end;
