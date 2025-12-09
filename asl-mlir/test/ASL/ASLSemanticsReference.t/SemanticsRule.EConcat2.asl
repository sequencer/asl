// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var T: boolean = ( '1111' :: '0000' ) == '11110000';

func main () => integer
begin
  return 0;
end;

