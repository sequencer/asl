// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

impdef func Foo{N: integer{32,64}}(
  n : boolean,
  mask : bits(64) { [0] lsb }) => bits(N)
begin
  return Zeros{N};
end;

impdef func Bar(n : integer) => integer
begin
  return n;
end;

implementation func Foo{N: integer{32,64}}(
  n : boolean,
  mask : bits(64) { [0] lsb }) => bits(N)
begin
  return Ones{N};
end;

func main() => integer
begin
  let res = Foo{32}(TRUE, Ones{64});
  return 0;
end;
