// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Real(x: integer) => real
begin
    return x * 1.0;
end;

pure func RoundDown(x: real) => integer
begin
    let round = RoundTowardsZero(x);

    if x >= 0.0 || x == Real(round) then
        return round;
    else
        return round - 1;
    end;
end;

pure func RoundTowardsZero(x: real) => integer
begin
    let x_pos = Abs(x);

    if x_pos < 1.0 then
        return 0;
    end;

    let log = ILog2(x_pos);
    var acc : integer = 2^log;

    for i=log-1 downto 0 do
        let next = acc + 2^i;
        if x_pos >= Real(next) then
          acc = next;
        end;
    end;

    return if x < 0.0 then -acc else acc;
end;

pure func Abs(x: real) => real
begin
  return if x >= 0.0 then x else -x;
end;

pure func ILog2(value : real) => integer
begin
    assert value != 0.0;
    var val : real = Abs(value);
    var low : integer;
    var high : integer;

    // Exponential search to find upper/lower power-of-2 exponent range
    if val >= 1.0 then
        low = 0; high = 1;
        while 2.0 ^ high <= val looplimit 2^128 do
            low = high;
            high = high * 2;
        end;
    else
        low = -1; high = 0;
        while 2.0 ^ low > val looplimit 2^128 do
            high = low;
            low = low * 2;
        end;
    end;

    // Binary search between low and high
    while low + 1 < high looplimit 2^128 do
        var mid = (low + high) DIVRM 2;
        if 2.0 ^ mid > val then
            high = mid;
        else
            low = mid;
        end;
    end;

    return low;
end;

type MyType of real; // An alias of real

func circle_circumference(radius: real) => real
begin
  let pi = 3.141592;
  return 2.0 * pi * radius;
end;

func main() => integer
begin
  var x: real = Real(5);
  x = circle_circumference(x as real);
  assert x as real == x;
  let y: integer = RoundDown(x);
  return 0;
end;
