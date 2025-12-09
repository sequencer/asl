// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  assert BitCount ('000') == 0;
  assert BitCount ('101') == 2;
  assert BitCount ('010') == 1;
  assert BitCount ('') == 0;

  assert LowestSetBit ('000') == 3;
  assert LowestSetBit ('101') == 0;
  assert LowestSetBit ('010') == 1;
  assert LowestSetBit ('') == 0;

  assert HighestSetBit ('000') == -1;
  assert HighestSetBit ('101') == 2;
  assert HighestSetBit ('010') == 1;
  assert HighestSetBit ('') == -1;

  return 0;
end;
