(******************************************************************************)
(*                            ASL JSON Backend                                *)
(******************************************************************************)
(*
 * SPDX-FileCopyrightText: Copyright 2025 Jiuyang Liu
 * SPDX-License-Identifier: Apache-2.0
 *)
(******************************************************************************)

open Asllib.AST
open Yojson.Safe

(** Utility functions *)
let position_to_json (pos : Lexing.position) : t =
  `Assoc [
    ("filename", `String pos.pos_fname);
    ("line", `Int pos.pos_lnum);
    ("column", `Int (pos.pos_cnum - pos.pos_bol + 1));
  ]

let version_to_json (v : version) : t =
  match v with
  | V0 -> `String "V0"
  | V1 -> `String "V1"

let annotated_to_json (conv : 'a -> t) (annotated : 'a annotated) : t =
  let base_json = conv annotated.desc in
  match base_json with
  | `Assoc fields ->
      `Assoc (fields @ [
        ("pos_start", position_to_json annotated.pos_start);
        ("pos_end", position_to_json annotated.pos_end);
        ("version", version_to_json annotated.version);
      ])
  | other ->
      `Assoc [
        ("desc", other);
        ("pos_start", position_to_json annotated.pos_start);
        ("pos_end", position_to_json annotated.pos_end);
        ("version", version_to_json annotated.version);
      ]

(** Convert literals to JSON *)
let literal_to_json (lit : literal) : t =
  match lit with
  | L_Int z -> `Assoc [("type", `String "L_Int"); ("value", `String (Z.to_string z))]
  | L_Bool b -> `Assoc [("type", `String "L_Bool"); ("value", `Bool b)]
  | L_Real q -> `Assoc [("type", `String "L_Real"); ("value", `String (Q.to_string q))]
  | L_BitVector bv -> `Assoc [("type", `String "L_BitVector"); ("value", `String (Asllib.Bitvector.to_string bv))]
  | L_String s -> `Assoc [("type", `String "L_String"); ("value", `String s)]
  | L_Label l -> `Assoc [("type", `String "L_Label"); ("value", `String l)]

(** Convert unary operators to JSON *)
let unop_to_json (op : unop) : t =
  match op with
  | BNOT -> `String "BNOT"
  | NEG -> `String "NEG"
  | NOT -> `String "NOT"

(** Convert binary operators to JSON *)
let binop_to_json (op : binop) : t =
  match op with
  | `AND -> `String "AND"
  | `BAND -> `String "BAND"
  | `BEQ -> `String "BEQ"
  | `BOR -> `String "BOR"
  | `DIV -> `String "DIV"
  | `DIVRM -> `String "DIVRM"
  | `XOR -> `String "XOR"
  | `EQ_OP -> `String "EQ_OP"
  | `GT -> `String "GT"
  | `GEQ -> `String "GEQ"
  | `IMPL -> `String "IMPL"
  | `LT -> `String "LT"
  | `LEQ -> `String "LEQ"
  | `MOD -> `String "MOD"
  | `MINUS -> `String "MINUS"
  | `MUL -> `String "MUL"
  | `NEQ -> `String "NEQ"
  | `OR -> `String "OR"
  | `PLUS -> `String "PLUS"
  | `POW -> `String "POW"
  | `RDIV -> `String "RDIV"
  | `SHL -> `String "SHL"
  | `SHR -> `String "SHR"
  | `CONCAT -> `String "CONCAT"

(** Convert subprogram types to JSON *)
let subprogram_type_to_json (st : subprogram_type) : t =
  match st with
  | ST_Procedure -> `String "ST_Procedure"
  | ST_Function -> `String "ST_Function"
  | ST_Getter -> `String "ST_Getter"
  | ST_EmptyGetter -> `String "ST_EmptyGetter"
  | ST_Setter -> `String "ST_Setter"
  | ST_EmptySetter -> `String "ST_EmptySetter"

(** Forward declarations for mutual recursion *)
let rec expr_to_json (e : expr) : t = annotated_to_json expr_desc_to_json e
and pattern_to_json (p : pattern) : t = annotated_to_json pattern_desc_to_json p
and ty_to_json (ty : ty) : t = annotated_to_json type_desc_to_json ty
and stmt_to_json (s : stmt) : t = annotated_to_json stmt_desc_to_json s

(** Convert slices to JSON *)
and slice_to_json (s : slice) : t =
  match s with
  | Slice_Single e -> `Assoc [("type", `String "Slice_Single"); ("expr", expr_to_json e)]
  | Slice_Range (e1, e2) -> `Assoc [("type", `String "Slice_Range"); ("start", expr_to_json e1); ("end", expr_to_json e2)]
  | Slice_Length (e1, e2) -> `Assoc [("type", `String "Slice_Length"); ("start", expr_to_json e1); ("length", expr_to_json e2)]
  | Slice_Star (e1, e2) -> `Assoc [("type", `String "Slice_Star"); ("start", expr_to_json e1); ("step", expr_to_json e2)]

(** Convert function calls to JSON *)
and call_to_json (c : call) : t =
  `Assoc [
    ("name", `String c.name);
    ("params", `List (List.map expr_to_json c.params));
    ("args", `List (List.map expr_to_json c.args));
    ("call_type", subprogram_type_to_json c.call_type);
  ]

(** Convert expression descriptions to JSON *)
and expr_desc_to_json (e : expr_desc) : t =
  match e with
  | E_Literal lit -> `Assoc [("type", `String "E_Literal"); ("literal", literal_to_json lit)]
  | E_Var id -> `Assoc [("type", `String "E_Var"); ("name", `String id)]
  | E_ATC (e, ty) -> `Assoc [("type", `String "E_ATC"); ("expr", expr_to_json e); ("target_type", ty_to_json ty)]
  | E_Binop (op, e1, e2) -> `Assoc [("type", `String "E_Binop"); ("op", binop_to_json op); ("left", expr_to_json e1); ("right", expr_to_json e2)]
  | E_Unop (op, e) -> `Assoc [("type", `String "E_Unop"); ("op", unop_to_json op); ("expr", expr_to_json e)]
  | E_Call c -> `Assoc [("type", `String "E_Call"); ("call", call_to_json c)]
  | E_Slice (e, slices) -> `Assoc [("type", `String "E_Slice"); ("expr", expr_to_json e); ("slices", `List (List.map slice_to_json slices))]
  | E_Cond (e1, e2, e3) -> `Assoc [("type", `String "E_Cond"); ("condition", expr_to_json e1); ("then_expr", expr_to_json e2); ("else_expr", expr_to_json e3)]
  | E_GetArray (e1, e2) -> `Assoc [("type", `String "E_GetArray"); ("array", expr_to_json e1); ("index", expr_to_json e2)]
  | E_GetEnumArray (e1, e2) -> `Assoc [("type", `String "E_GetEnumArray"); ("array", expr_to_json e1); ("enum_value", expr_to_json e2)]
  | E_GetField (e, field) -> `Assoc [("type", `String "E_GetField"); ("expr", expr_to_json e); ("field", `String field)]
  | E_GetFields (e, fields) -> `Assoc [("type", `String "E_GetFields"); ("expr", expr_to_json e); ("fields", `List (List.map (fun f -> `String f) fields))]
  | E_GetCollectionFields (id, fields) -> `Assoc [("type", `String "E_GetCollectionFields"); ("collection", `String id); ("fields", `List (List.map (fun f -> `String f) fields))]
  | E_GetItem (e, i) -> `Assoc [("type", `String "E_GetItem"); ("expr", expr_to_json e); ("index", `Int i)]
  | E_Record (ty, fields) -> `Assoc [("type", `String "E_Record"); ("record_type", ty_to_json ty); ("fields", `List (List.map (fun (name, e) -> `Assoc [("name", `String name); ("value", expr_to_json e)]) fields))]
  | E_Tuple exprs -> `Assoc [("type", `String "E_Tuple"); ("elements", `List (List.map expr_to_json exprs))]
  | E_Array {length; value} -> `Assoc [("type", `String "E_Array"); ("length", expr_to_json length); ("value", expr_to_json value)]
  | E_EnumArray {enum; labels; value} -> `Assoc [("type", `String "E_EnumArray"); ("enum", `String enum); ("labels", `List (List.map (fun l -> `String l) labels)); ("value", expr_to_json value)]
  | E_Arbitrary ty -> `Assoc [("type", `String "E_Arbitrary"); ("target_type", ty_to_json ty)]
  | E_Pattern (e, p) -> `Assoc [("type", `String "E_Pattern"); ("expr", expr_to_json e); ("pattern", pattern_to_json p)]

(** Convert pattern descriptions to JSON *)
and pattern_desc_to_json (p : pattern_desc) : t =
  match p with
  | Pattern_All -> `Assoc [("type", `String "Pattern_All")]
  | Pattern_Any patterns -> `Assoc [("type", `String "Pattern_Any"); ("patterns", `List (List.map pattern_to_json patterns))]
  | Pattern_Geq e -> `Assoc [("type", `String "Pattern_Geq"); ("expr", expr_to_json e)]
  | Pattern_Leq e -> `Assoc [("type", `String "Pattern_Leq"); ("expr", expr_to_json e)]
  | Pattern_Mask mask -> `Assoc [("type", `String "Pattern_Mask"); ("mask", `String (Asllib.Bitvector.mask_to_string mask))]
  | Pattern_Not p -> `Assoc [("type", `String "Pattern_Not"); ("pattern", pattern_to_json p)]
  | Pattern_Range (e1, e2) -> `Assoc [("type", `String "Pattern_Range"); ("start", expr_to_json e1); ("end", expr_to_json e2)]
  | Pattern_Single e -> `Assoc [("type", `String "Pattern_Single"); ("expr", expr_to_json e)]
  | Pattern_Tuple patterns -> `Assoc [("type", `String "Pattern_Tuple"); ("patterns", `List (List.map pattern_to_json patterns))]

(** Convert integer constraints to JSON *)
and int_constraint_to_json (c : int_constraint) : t =
  match c with
  | Constraint_Exact e -> `Assoc [("type", `String "Constraint_Exact"); ("expr", expr_to_json e)]
  | Constraint_Range (e1, e2) -> `Assoc [("type", `String "Constraint_Range"); ("start", expr_to_json e1); ("end", expr_to_json e2)]

(** Convert constraint kinds to JSON *)
and constraint_kind_to_json (ck : constraint_kind) : t =
  match ck with
  | UnConstrained -> `Assoc [("type", `String "UnConstrained")]
  | WellConstrained (constraints, _precision_flag) -> `Assoc [("type", `String "WellConstrained"); ("constraints", `List (List.map int_constraint_to_json constraints))]
  | PendingConstrained -> `Assoc [("type", `String "PendingConstrained")]
  | Parameterized name -> `Assoc [("type", `String "Parameterized"); ("name", `String name)]

(** Convert bitfields to JSON *)
and bitfield_to_json (bf : bitfield) : t =
  match bf with
  | BitField_Simple (name, slices) -> `Assoc [("type", `String "BitField_Simple"); ("name", `String name); ("slices", `List (List.map slice_to_json slices))]
  | BitField_Nested (name, slices, nested) -> `Assoc [("type", `String "BitField_Nested"); ("name", `String name); ("slices", `List (List.map slice_to_json slices)); ("nested", `List (List.map bitfield_to_json nested))]
  | BitField_Type (name, slices, ty) -> `Assoc [("type", `String "BitField_Type"); ("name", `String name); ("slices", `List (List.map slice_to_json slices)); ("field_type", ty_to_json ty)]

(** Convert array indexes to JSON *)
and array_index_to_json (ai : array_index) : t =
  match ai with
  | ArrayLength_Expr e -> `Assoc [("type", `String "ArrayLength_Expr"); ("expr", expr_to_json e)]
  | ArrayLength_Enum (enum, labels) -> `Assoc [("type", `String "ArrayLength_Enum"); ("enum", `String enum); ("labels", `List (List.map (fun l -> `String l) labels))]

(** Convert type descriptions to JSON *)
and type_desc_to_json (td : type_desc) : t =
  match td with
  | T_Int ck -> `Assoc [("type", `String "T_Int"); ("constraint_kind", constraint_kind_to_json ck)]
  | T_Bits (e, bitfields) -> `Assoc [("type", `String "T_Bits"); ("width", expr_to_json e); ("bitfields", `List (List.map bitfield_to_json bitfields))]
  | T_Real -> `Assoc [("type", `String "T_Real")]
  | T_String -> `Assoc [("type", `String "T_String")]
  | T_Bool -> `Assoc [("type", `String "T_Bool")]
  | T_Enum labels -> `Assoc [("type", `String "T_Enum"); ("labels", `List (List.map (fun l -> `String l) labels))]
  | T_Tuple types -> `Assoc [("type", `String "T_Tuple"); ("types", `List (List.map ty_to_json types))]
  | T_Array (ai, ty) -> `Assoc [("type", `String "T_Array"); ("index", array_index_to_json ai); ("element_type", ty_to_json ty)]
  | T_Record fields -> `Assoc [("type", `String "T_Record"); ("fields", `List (List.map (fun (name, ty) -> `Assoc [("name", `String name); ("field_type", ty_to_json ty)]) fields))]
  | T_Exception fields -> `Assoc [("type", `String "T_Exception"); ("fields", `List (List.map (fun (name, ty) -> `Assoc [("name", `String name); ("field_type", ty_to_json ty)]) fields))]
  | T_Collection fields -> `Assoc [("type", `String "T_Collection"); ("fields", `List (List.map (fun (name, ty) -> `Assoc [("name", `String name); ("field_type", ty_to_json ty)]) fields))]
  | T_Named name -> `Assoc [("type", `String "T_Named"); ("name", `String name)]

(** Convert left-hand side expressions to JSON *)
and lexpr_to_json (le : lexpr) : t = annotated_to_json lexpr_desc_to_json le

and lexpr_desc_to_json (led : lexpr_desc) : t =
  match led with
  | LE_Discard -> `Assoc [("type", `String "LE_Discard")]
  | LE_Var name -> `Assoc [("type", `String "LE_Var"); ("name", `String name)]
  | LE_Slice (le, slices) -> `Assoc [("type", `String "LE_Slice"); ("lexpr", lexpr_to_json le); ("slices", `List (List.map slice_to_json slices))]
  | LE_SetArray (le, e) -> `Assoc [("type", `String "LE_SetArray"); ("lexpr", lexpr_to_json le); ("index", expr_to_json e)]
  | LE_SetEnumArray (le, e) -> `Assoc [("type", `String "LE_SetEnumArray"); ("lexpr", lexpr_to_json le); ("index", expr_to_json e)]
  | LE_SetField (le, field) -> `Assoc [("type", `String "LE_SetField"); ("lexpr", lexpr_to_json le); ("field", `String field)]
  | LE_SetFields (le, fields, type_info) -> `Assoc [("type", `String "LE_SetFields"); ("lexpr", lexpr_to_json le); ("fields", `List (List.map (fun f -> `String f) fields)); ("type_info", `List (List.map (fun (a, b) -> `List [`Int a; `Int b]) type_info))]
  | LE_SetCollectionFields (id, fields, type_info) -> `Assoc [("type", `String "LE_SetCollectionFields"); ("collection", `String id); ("fields", `List (List.map (fun f -> `String f) fields)); ("type_info", `List (List.map (fun (a, b) -> `List [`Int a; `Int b]) type_info))]
  | LE_Destructuring exprs -> `Assoc [("type", `String "LE_Destructuring"); ("lexprs", `List (List.map lexpr_to_json exprs))]

(** Convert declaration keywords to JSON *)
and local_decl_keyword_to_json (ldk : local_decl_keyword) : t =
  match ldk with
  | LDK_Var -> `String "LDK_Var"
  | LDK_Constant -> `String "LDK_Constant"
  | LDK_Let -> `String "LDK_Let"

and local_decl_item_to_json (ldi : local_decl_item) : t =
  match ldi with
  | LDI_Var name -> `Assoc [("type", `String "LDI_Var"); ("name", `String name)]
  | LDI_Tuple names -> `Assoc [("type", `String "LDI_Tuple"); ("names", `List (List.map (fun n -> `String n) names))]

(** Convert for directions to JSON *)
and for_direction_to_json (fd : for_direction) : t =
  match fd with
  | Up -> `String "Up"
  | Down -> `String "Down"

(** Convert catchers to JSON *)
and catcher_to_json ((name_opt, ty, stmt) : catcher) : t =
  `Assoc [
    ("name", Option.fold ~none:`Null ~some:(fun n -> `String n) name_opt);
    ("exception_type", ty_to_json ty);
    ("stmt", stmt_to_json stmt);
  ]

(** Convert statement descriptions to JSON *)
and stmt_desc_to_json (sd : stmt_desc) : t =
  match sd with
  | S_Pass -> `Assoc [("type", `String "S_Pass")]
  | S_Seq (s1, s2) -> `Assoc [("type", `String "S_Seq"); ("first", stmt_to_json s1); ("second", stmt_to_json s2)]
  | S_Decl (kw, ldi, ty_opt, e_opt) -> `Assoc [("type", `String "S_Decl"); ("keyword", local_decl_keyword_to_json kw); ("decl_item", local_decl_item_to_json ldi); ("type", Option.fold ~none:`Null ~some:ty_to_json ty_opt); ("init", Option.fold ~none:`Null ~some:expr_to_json e_opt)]
  | S_Assign (le, e) -> `Assoc [("type", `String "S_Assign"); ("lhs", lexpr_to_json le); ("rhs", expr_to_json e)]
  | S_Call c -> `Assoc [("type", `String "S_Call"); ("call", call_to_json c)]
  | S_Return e_opt -> `Assoc [("type", `String "S_Return"); ("expr", Option.fold ~none:`Null ~some:expr_to_json e_opt)]
  | S_Cond (e, s1, s2) -> `Assoc [("type", `String "S_Cond"); ("condition", expr_to_json e); ("then_stmt", stmt_to_json s1); ("else_stmt", stmt_to_json s2)]
  | S_Assert e -> `Assoc [("type", `String "S_Assert"); ("expr", expr_to_json e)]
  | S_For {index_name; start_e; dir; end_e; body; limit} -> `Assoc [("type", `String "S_For"); ("index_name", `String index_name); ("start", expr_to_json start_e); ("direction", for_direction_to_json dir); ("end", expr_to_json end_e); ("body", stmt_to_json body); ("limit", Option.fold ~none:`Null ~some:expr_to_json limit)]
  | S_While (e, limit_opt, s) -> `Assoc [("type", `String "S_While"); ("condition", expr_to_json e); ("limit", Option.fold ~none:`Null ~some:expr_to_json limit_opt); ("body", stmt_to_json s)]
  | S_Repeat (s, e, limit_opt) -> `Assoc [("type", `String "S_Repeat"); ("body", stmt_to_json s); ("condition", expr_to_json e); ("limit", Option.fold ~none:`Null ~some:expr_to_json limit_opt)]
  | S_Throw expr_ty_opt -> `Assoc [("type", `String "S_Throw"); ("value", Option.fold ~none:`Null ~some:(fun (e, ty_opt) -> `Assoc [("expr", expr_to_json e); ("type", Option.fold ~none:`Null ~some:ty_to_json ty_opt)]) expr_ty_opt)]
  | S_Try (s, catchers, otherwise_opt) -> `Assoc [("type", `String "S_Try"); ("body", stmt_to_json s); ("catchers", `List (List.map catcher_to_json catchers)); ("otherwise", Option.fold ~none:`Null ~some:stmt_to_json otherwise_opt)]
  | S_Print {args; newline; debug} -> `Assoc [("type", `String "S_Print"); ("args", `List (List.map expr_to_json args)); ("newline", `Bool newline); ("debug", `Bool debug)]
  | S_Unreachable -> `Assoc [("type", `String "S_Unreachable")]
  | S_Pragma (name, args) -> `Assoc [("type", `String "S_Pragma"); ("name", `String name); ("args", `List (List.map expr_to_json args))]

(** Convert function qualifiers to JSON *)
let func_qualifier_to_json (fq : func_qualifier) : t =
  match fq with
  | Pure -> `String "Pure"
  | Readonly -> `String "Readonly"
  | Noreturn -> `String "Noreturn"

(** Convert override info to JSON *)
let override_info_to_json (oi : override_info) : t =
  match oi with
  | Impdef -> `String "Impdef"
  | Implementation -> `String "Implementation"

(** Convert subprogram bodies to JSON *)
let subprogram_body_to_json (sb : subprogram_body) : t =
  match sb with
  | SB_ASL s -> `Assoc [("type", `String "SB_ASL"); ("stmt", stmt_to_json s)]
  | SB_Primitive side_effecting -> `Assoc [("type", `String "SB_Primitive"); ("side_effecting", `Bool side_effecting)]

(** Convert function definitions to JSON *)
let func_to_json (f : func) : t =
  `Assoc [
    ("name", `String f.name);
    ("parameters", `List (List.map (fun (name, ty_opt) -> `Assoc [("name", `String name); ("type", Option.fold ~none:`Null ~some:ty_to_json ty_opt)]) f.parameters));
    ("args", `List (List.map (fun (name, ty) -> `Assoc [("name", `String name); ("type", ty_to_json ty)]) f.args));
    ("body", subprogram_body_to_json f.body);
    ("return_type", Option.fold ~none:`Null ~some:ty_to_json f.return_type);
    ("subprogram_type", subprogram_type_to_json f.subprogram_type);
    ("recurse_limit", Option.fold ~none:`Null ~some:expr_to_json f.recurse_limit);
    ("qualifier", Option.fold ~none:`Null ~some:func_qualifier_to_json f.qualifier);
    ("override", Option.fold ~none:`Null ~some:override_info_to_json f.override);
    ("builtin", `Bool f.builtin);
  ]

(** Convert global declaration keywords to JSON *)
let global_decl_keyword_to_json (gdk : global_decl_keyword) : t =
  match gdk with
  | GDK_Constant -> `String "GDK_Constant"
  | GDK_Config -> `String "GDK_Config"
  | GDK_Let -> `String "GDK_Let"
  | GDK_Var -> `String "GDK_Var"

(** Convert global declarations to JSON *)
let global_decl_to_json (gd : global_decl) : t =
  `Assoc [
    ("keyword", global_decl_keyword_to_json gd.keyword);
    ("name", `String gd.name);
    ("type", Option.fold ~none:`Null ~some:ty_to_json gd.ty);
    ("initial_value", Option.fold ~none:`Null ~some:expr_to_json gd.initial_value);
  ]

(** Convert declaration descriptions to JSON *)
let decl_desc_to_json (dd : decl_desc) : t =
  match dd with
  | D_Func f -> `Assoc [("type", `String "D_Func"); ("func", func_to_json f)]
  | D_GlobalStorage gd -> `Assoc [("type", `String "D_GlobalStorage"); ("global_decl", global_decl_to_json gd)]
  | D_TypeDecl (name, ty, deps_opt) -> 
      let deps_json = match deps_opt with
        | None -> `Null
        | Some (enum_name, fields) -> 
            `Assoc [
              ("enum", `String enum_name);
              ("fields", `List (List.map (fun (name, ty) -> `Assoc [("name", `String name); ("type", ty_to_json ty)]) fields))
            ]
      in
      `Assoc [("type", `String "D_TypeDecl"); ("name", `String name); ("type_def", ty_to_json ty); ("dependencies", deps_json)]
  | D_Pragma (name, args) -> `Assoc [("type", `String "D_Pragma"); ("name", `String name); ("args", `List (List.map expr_to_json args))]

(** Convert declarations to JSON *)
let decl_to_json (d : decl) : t = annotated_to_json decl_desc_to_json d

(** Convert complete AST to JSON *)
let ast_to_json (ast : Asllib.AST.t) : Yojson.Safe.t =
  `Assoc [
    ("type", `String "ASL_AST");
    ("declarations", `List (List.map decl_to_json ast));
  ]

(** Parser configuration for JSON backend *)
module ParserConfig : Asllib.ParserConfig.CONFIG = struct
  let allow_no_end_semicolon = true
  let allow_expression_elsif = true 
  let allow_storage_discards = true
  let allow_hyphenated_pending_constraint = true
  let allow_local_constants = true
  let allow_empty_structured_type_declarations = true
  let allow_function_like_statements = true
end

module LexerConfig : Asllib.Lexer.CONFIG = struct
  let allow_double_underscore = true
  let allow_unknown = true
  let allow_single_arrows = true
  let allow_function_like_statements = true
end

module Parser = Asllib.Parser.Make(ParserConfig)
module Lexer = Asllib.Lexer.Make(LexerConfig)

(** Typing configuration for JSON backend *)
module TypingConfig : Asllib.Typing.ANNOTATE_CONFIG = struct
  let check = Asllib.Typing.TypeCheck
  let output_format = Asllib.Error.CSV
  let print_typed = true
  let use_field_getter_extension = true
  let fine_grained_side_effects = true
  let use_conflicting_side_effects_extension = true
  let override_mode = Asllib.Typing.Permissive
  let control_flow_analysis = true
end

module Typing = Asllib.Typing.Annotate(TypingConfig)

(** Parse a file and convert to JSON *)
let parse_file_to_json ?(with_stdlib=true) (filename : string) : Yojson.Safe.t =
  let ic = open_in filename in
  let lexbuf = Lexing.from_channel ic in
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = filename };
  try
    let ast = Parser.spec Lexer.token lexbuf in
    let ast_to_type = if with_stdlib then Asllib.Builder.with_stdlib ast else ast in
    let typed_ast, _env = Typing.type_check_ast ast_to_type in
    close_in ic;
    ast_to_json typed_ast
  with
  | e ->
      close_in ic;
      raise e

(** Parse a string and convert to JSON *)
let parse_string_to_json ?(with_stdlib=true) (input : string) : Yojson.Safe.t =
  let lexbuf = Lexing.from_string input in
  let ast = Parser.spec Lexer.token lexbuf in
  let ast_to_type = if with_stdlib then Asllib.Builder.with_stdlib ast else ast in
  let typed_ast, _env = Typing.type_check_ast ast_to_type in
  ast_to_json typed_ast

(** Main function for the executable *)
let () =
  let usage_msg = Printf.sprintf "Usage: %s [--no-std] <asl_file>" Sys.argv.(0) in
  let with_stdlib = ref true in
  let filename = ref "" in
  
  let spec = [
    ("--no-std", Arg.Clear with_stdlib, " Parse AST without stdlib");
  ] in
  
  let set_filename f = 
    if !filename = "" then filename := f 
    else (Printf.eprintf "Error: Multiple files specified\n"; exit 1) in
  
  Arg.parse spec set_filename usage_msg;
  
  if !filename = "" then (
    Printf.eprintf "%s\n" usage_msg;
    exit 1
  ) else (
    try
      let json = parse_file_to_json ~with_stdlib:(!with_stdlib) !filename in
      print_endline (Yojson.Safe.to_string json)
    with
    | Sys_error msg ->
        Printf.eprintf "Error: %s\n" msg;
        exit 1
    | e ->
        Printf.eprintf "Parse error: %s\n" (Printexc.to_string e);
        exit 1
  )
