// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

// CHECK: uint8_t bytes[16]
var bytes: array [[16]] of bits(8);

// type Coord of enumeration {X, Y, Z};

// var position: array [[Coord]] of integer;

// var points: array [[4]] of Point;

// var matrix: array [[3]] of array [[3]] of integer;
