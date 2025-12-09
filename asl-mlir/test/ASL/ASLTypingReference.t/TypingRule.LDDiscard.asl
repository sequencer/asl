// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let - = 42;
  var - = 42;
  constant - = 42;
  let - = "abc";
  let - = '101010';

  return 0;
end;
