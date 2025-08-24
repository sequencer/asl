// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: module {

type SuperEnum of enumeration {LOW, HIGH};
// LOW and HIGH are of type enumeration {LOW, HIGH}
type SubEnum subtypes SuperEnum; // Legal
type OtherEnum of SuperEnum; // Legal
