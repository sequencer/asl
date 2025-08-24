// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type pagros of integer{1,2,4,8,16};

var ones = Ones{64};

func f{sz:pagros}() => bits(sz)
begin
  return ones[sz-1:0];
end;

func main() => integer
begin
  let x = f{8};
  println x;
  return 0;
end;
