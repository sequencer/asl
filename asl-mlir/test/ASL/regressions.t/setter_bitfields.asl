// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyBV of bits(8) { [5] bitfield };

accessor F() <=> v: MyBV
begin
  readonly getter
    return Ones{8} as MyBV;
  end;

  setter
    assert v.bitfield == '0';
  end;
end;

func main () => integer
begin
  let res = F().bitfield;
  assert res == '1';
  F().bitfield = '0';

  return 0;
end;
