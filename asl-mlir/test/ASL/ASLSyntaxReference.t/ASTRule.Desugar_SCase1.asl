// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var x : integer = ARBITRARY: integer;
  // The following case statement:
  case x of
    when 42 => x = 42;
    when <= 42 => x = 0;
    otherwise => x = 43;
  end;
  // can be desugared into the following condition statement:
  if x IN {42} then
    x = 42;
  else
    if x IN {<= 42} then x = 0; else x = 43; end;
  end;

  return x;
end;