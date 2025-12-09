// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

var myData: bits(16) {
    [4] flag,
    [3:0, 8:5] data,
    [9:0] value
};
