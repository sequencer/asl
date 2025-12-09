// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

constant FOUR = 4;
constant FIVE = 4;

var myData: bits(16) {
    [FOUR] flag,            // { Other(FOUR) }
    [3:0, 8:FIVE] data {    // { Other(FIVE), Other(FOUR) }
        [FOUR] data_5       // { Other(FOUR) }
    },
    [9:0] value             // { }
};
