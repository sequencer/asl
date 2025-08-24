// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Pair of (integer, boolean);

type T of array [[3]] of real;
type Coord of enumeration { CX, CY, CZ };
type PointArray of array [[Coord]] of real;

type PointRecord of record
  { x : real, y : real, z : real };

func main () => integer
begin
  let p = (0, FALSE);

  var t1 : T; var t2 : PointArray;
  t1[[0]] = t2[[CX]];

  let o = PointRecord { x=0.0, y=0.0, z=0.0 };
  t2[[CZ]] = o.z;

  return 0;
end;
