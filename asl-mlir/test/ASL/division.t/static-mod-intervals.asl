// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func foo ()
begin
  let a = ARBITRARY: integer  {10..20};
  let b = ARBITRARY: integer {a};
  let c = ARBITRARY: integer {-1000..1000};

  let d: integer {0..20-1} = c MOD a;
  let e: integer {0..a-1} = c MOD b;
end;
