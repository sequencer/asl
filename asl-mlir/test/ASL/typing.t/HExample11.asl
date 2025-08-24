// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func min_highest_set_bit_example{N}(curr : bits(N))
begin
    var highest = HighestSetBit(curr) as integer{0..7};
    var minimum = Min(highest, 7) as integer{0..7};

    let size = minimum;
    var x = Zeros {N};
end;
