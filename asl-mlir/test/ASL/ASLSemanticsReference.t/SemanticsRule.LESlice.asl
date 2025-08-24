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

func main () => integer
begin
  var x = '11 11 1111';
  x[3:0, 7:6] = '000000';
  assert x == '00110000';

  x[(1)*:(2)] = Ones{2};
  assert x == '00111100';

  return 0;
end;
