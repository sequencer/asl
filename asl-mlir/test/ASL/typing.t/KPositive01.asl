// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type ty of integer{-1..2};
let w : ty = 2;
var x: bits(w) = '11'; 
