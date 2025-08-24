// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

var a = 42;
let b = 42;
constant c = 42;
config d : integer = 42;
var e, f, g : integer;
