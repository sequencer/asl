// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var global_x: integer = 3;

func main () => integer
  begin

    assert global_x == 3;
    return 0;

  end;
