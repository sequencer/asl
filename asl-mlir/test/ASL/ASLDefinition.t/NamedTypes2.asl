// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

type qualifiedData of bits(16) {
    [4] flag,
    [3: 0, 8:5] data,
    [9:0] value
};

type DatawithFlag of qualifiedData;
