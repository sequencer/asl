// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
  begin
    assert (0.0 ^ 0 == 1.0);

    return 0;
  end;

