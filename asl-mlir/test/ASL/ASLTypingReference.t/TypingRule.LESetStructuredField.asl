// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

type MyRecord of record {status: boolean, time: integer, data: bits(8)};

func main() => integer
begin
    var x : MyRecord;
    x.status = TRUE;
    x.time = 0;
    x.data = Ones{8};
    return 0;
end;
