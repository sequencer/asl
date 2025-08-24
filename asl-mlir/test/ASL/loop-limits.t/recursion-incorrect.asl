// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func recurse (n: integer) => integer
recurselimit 5
begin
  println (n);
  if n >= 10 then return 1;
  else return 1 + recurse (n+1); end;
end;

func main () => integer
begin
  println "Number of calls: ", recurse (0);
  println "Number of calls: ", recurse (0);

  return 0;
end;

