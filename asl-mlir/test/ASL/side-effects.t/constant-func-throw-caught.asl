// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type E of exception {-};

pure func foo (x: integer) => integer
begin
  try
    throw E {-};
  catch
    when E => return 19;
  end;
end;

constant C = foo (4);

func main () => integer
begin
  assert C == 19;

  return 0;
end;



