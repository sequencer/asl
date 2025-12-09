// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

// Global storage declarations  Side effects
constant c = 10;                // GlobalEffect(SE_Pure), Immutability(TRUE)
config cfg : integer = 20;      // GlobalEffect(SE_Readonly), Immutability(TRUE)
let l = 30;                     // GlobalEffect(SE_Readonly), Immutability(TRUE)
var v = 40;                     // GlobalEffect(SE_Readonly), Immutability(FALSE)
