// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

constant z = Zeros{15};

func main () => integer
begin
  assert z == '000000000000000';

  return 0;
end;
