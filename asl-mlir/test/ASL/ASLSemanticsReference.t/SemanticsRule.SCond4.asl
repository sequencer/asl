// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var size:bits(2);
  var esize:integer;
  var elements:integer;

  if size == '01' then
    esize = 16;
    elements = 4;
  end;

  return 0;
end;
