// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func foo() // GlobalEffect(SE_Pure), Immutability(TRUE)
begin
  pass;
end;

readonly func bar() // GlobalEffect(SE_Readonly), Immutability(FALSE)
begin
  pass;
end;

noreturn func goo() // GlobalEffect(SE_Impure), Immutability(FALSE)
begin
  pass;
end;

func baz() // GlobalEffect(SE_Impure), Immutability(FALSE)
begin
  pass;
end;
