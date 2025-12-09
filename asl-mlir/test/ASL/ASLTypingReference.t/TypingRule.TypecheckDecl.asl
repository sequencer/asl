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

type MyRecord of record {
    high_bits: bits(32),
    low_bits: bits(32),
};

type MyException of exception {
    msg: string,
};


var rec: MyRecord;
var exc: MyException;

var coll: collection {
    high_bits: bits(32),
    low_bits: bits(32),
};

accessor Rec() <=> values: bits (64)
begin
  readonly getter
        return rec.high_bits :: rec.low_bits;
    end;

    setter
        rec.high_bits = values[63:32];
        rec.low_bits = values[31:0];
    end;
end;

func main() => integer
begin
    println Rec();
    Rec() = Ones{64};
    println Rec();
    return 0;
end;
