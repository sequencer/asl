// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type FlaggedPacket of bits(8) {
    [7:1] data {
        [2:0] low,
        [6:3] high
    },
    [0] flag
};

func main() => integer
begin
    var y: FlaggedPacket;
    var x: bits(8) {[0] flag, [7:1] data { [2:0] low }} = y;
    return 0;
end;
