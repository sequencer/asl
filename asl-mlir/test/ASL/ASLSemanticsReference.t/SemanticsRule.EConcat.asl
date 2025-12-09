// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let x = '10' :: '11';
  assert x=='1011';

  let y = '' :: '';
  assert y == '';

  return 0;
end;
