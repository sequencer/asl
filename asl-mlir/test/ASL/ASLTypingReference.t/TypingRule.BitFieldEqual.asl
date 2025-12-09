// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func main() => integer
begin
    var x : bits(64) { [16+:16] data } = Zeros{64} as bits(64) { [31:16] data };

    var y : bits(64) { [16+:16] data { [0] lsb } } =  Zeros{64} as
            bits(64) { [31:16] data { [0] lsb } };

    var z : bits(64) { [0] lsb : bits(1) } = Zeros{64} as bits(64) { [0] lsb : bit };
    return 0;
end;
