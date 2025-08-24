// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Packet of bits(8) {
    [0] flag,
    [7:1] data,
    [7:1] detailed_data {
        [6:3] info : bits(4) {
            [0] info_0
        },
        [2:0] crc
    }
};

func main() => integer
begin
    var x = '1010 0101';
    var y = '1111 0011';
    - = x AND y;
    return 0;
end;
