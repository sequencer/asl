// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  if TRUE then
    assert TRUE;
  else
    assert FALSE;
  end;

  return 0;
end;
