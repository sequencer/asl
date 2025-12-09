// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Color of enumeration { GREEN, ORANGE, RED };

func rotate_color(c : Color) => Color
begin
    case c of
        when GREEN => return ORANGE;
        when ORANGE => return RED;
        when RED => return GREEN;
    end;
end;

// Legal: subprograms and enumeration labels exist
// in separate namespaces.
func GREEN() => integer
begin
    return 0;
end;
