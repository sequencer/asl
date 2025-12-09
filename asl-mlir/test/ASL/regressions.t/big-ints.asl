// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  assert (0x12345678123456789 > 0);
  // assert (UInt(0x2a2345678123456789[127:64]) == 42);
  
  return 0;
end;
