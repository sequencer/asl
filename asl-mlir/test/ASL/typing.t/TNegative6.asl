// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

////////////////////////////////////////////////////////////////////////////////
// comparisons
////////////////////////////////////////////////////////////////////////////////

func foo() => integer {8, 16}
begin
  return ARBITRARY : integer {8, 16};
end;

func negative6()
begin
    // Same principle applies to the when clauses in a case statement
    let testG : integer {8,16} = foo();
    case testG of
            when 32 =>
                pass;
                // <some code>
            when 64 =>
                pass;
                // <some code>
    end;
end;
