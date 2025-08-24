// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type E of exception {-};

func always_throws () => integer
begin
  throw E {-};
end;

func main () => integer
begin
  var y: integer = 0;
  try
    let x = always_throws ();
  catch
    when E => y = 42;
  end;

  assert y == 42;

  return 0;
end;
