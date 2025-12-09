// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

var myData: bits(16) {
    [4] flag,
    [3:0, 5+:3] data,
    [3*:4] value
};
