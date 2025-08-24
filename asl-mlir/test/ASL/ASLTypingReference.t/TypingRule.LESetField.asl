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

type Message of bits(25) {
    [0] status,
    [16:1] time : bits(16) {
        [0] odd_even
    },
    [24:17] data
};

func main() => integer
begin
    var x : Message;
    x.status = '1';
    x.time = Zeros{16};
    x.time.odd_even = '1';
    x.data = Ones{8};
    assert x == '11111111 0000000000000001 1';
    return 0;
end;
