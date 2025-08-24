// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var x: integer;
  var y: integer;

  if x > y then
      return 1;
  elsif x < y then
      return -1;
  else
      return 0;
  end;
end;
