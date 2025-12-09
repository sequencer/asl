// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let - = 42;
  var - = TRUE;
  constant - = "abc";
  let (-, -, -) = ('1', 1.0, 1);

  return 0;
end;

