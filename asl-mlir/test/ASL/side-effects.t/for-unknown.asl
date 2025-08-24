// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

readonly func unknown () => integer {0..10}
begin
  return ARBITRARY: integer {0..10};
end;

func main () => integer
begin
  for i = 0 to unknown () do
    pass;
  end;

  return 0;
end;

