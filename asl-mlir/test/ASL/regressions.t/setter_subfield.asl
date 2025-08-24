// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyBV of bits(8) { [5] bitfield };


accessor F() <=> v: MyBV
begin
  readonly getter
    return Zeros{8} as MyBV;
  end;

  setter
    assert v[0] == '0';
  end;
end;

func main () => integer
begin
  F().bitfield = '0';

  return 0;
end;
