// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

type Nested_Type of bits(2) {
    [1:0] sub {
        [1:0] sub {
            [1,0] lowest
        }
    },

    [1:0] lowest
};
