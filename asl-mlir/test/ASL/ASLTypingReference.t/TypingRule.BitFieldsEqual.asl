// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

impdef          func Foo(bv : bits(2) { [0] lsb, [1] msb }) begin pass; end;
implementation  func Foo(bv : bits(2) { [0] lsb, [1] msb }) begin pass; end;
