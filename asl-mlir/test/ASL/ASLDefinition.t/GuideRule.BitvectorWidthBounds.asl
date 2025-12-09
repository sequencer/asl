// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

var large_bitvector: bits(2^20);
var zero_width_bitvector: bits(0);
