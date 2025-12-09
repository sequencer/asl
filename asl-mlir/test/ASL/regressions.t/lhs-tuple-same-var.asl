// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type BV of bits (8) {
  [1:0] fld
};

func main() => integer
begin
  var bv : BV;
  (bv[7], -, bv.fld, bv.fld) = ('1', TRUE, '00', '11');
  assert bv.fld == '11';

  return 0;
end;
