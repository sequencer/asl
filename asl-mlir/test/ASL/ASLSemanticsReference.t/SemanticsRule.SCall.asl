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

var g: bits(7);

func catenate_into_g{N, M}(x: bits(N), y: bits(M), order: boolean)
begin
    if order then
        g = (x :: y) as bits(7);
    else
        g = (y :: x) as bits(7);
    end;
end;

func zero() => integer
begin
  return 0;
end;

func main() => integer
begin
    var x = '1101';
    var y = Ones{3};
    assert g == Zeros{7};
    catenate_into_g{4, 3}(x, y, TRUE);
    assert g == '1101 111';

    - = zero();
    // The following statement in comment is illegal as 'zero'
    // a function, not a procedure, and its returned value
    // must be consumed.
    // zero();
    return 0;
end;
