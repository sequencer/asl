// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

type superInt   of integer;
type subInt     of integer subtypes superInt;
type subsubInt  of integer subtypes subInt;
type otherInt   of integer;
