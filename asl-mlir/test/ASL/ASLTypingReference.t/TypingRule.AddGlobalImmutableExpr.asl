// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {
let w: integer{1..2} = 2;

// The static environment remembers w = 2.
var x: bits(w) = '11';



