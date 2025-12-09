// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({


type a_record_ty of record {
    flag : boolean,
    count: integer,
    data : bit
};

type aa_record_ty of record {
    flag : boolean,
    count: integer,
    data : bit,
    foo: integer
} subtypes a_record_ty;

readonly func equal_a_record_ty (x: a_record_ty, y: a_record_ty) => boolean
begin
  return
    x.flag == y.flag &&
    x.count == y.count &&
    x.data == y.data;
end;

func main() => integer
begin
  let a = a_record_ty {
    flag = TRUE,
    count = 3,
    data = '1'
  };
  let aa = aa_record_ty {
    flag = TRUE,
    count = 3,
    data = '1',
    foo = 5
  };
  assert equal_a_record_ty (a, aa);

  return 0;
end;


