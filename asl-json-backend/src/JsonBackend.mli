(******************************************************************************)
(*                            ASL JSON Backend                                *)
(******************************************************************************)
(*
 * SPDX-FileCopyrightText: Copyright 2025 Jiuyang Liu
 * SPDX-License-Identifier: Apache-2.0
 *)
(******************************************************************************)

(** JSON encoder for ASL AST types *)

(** Convert an ASL literal to JSON *)
val literal_to_json : Asllib.AST.literal -> Yojson.Safe.t

(** Convert an ASL expression to JSON *)
val expr_to_json : Asllib.AST.expr -> Yojson.Safe.t

(** Convert an ASL statement to JSON *)
val stmt_to_json : Asllib.AST.stmt -> Yojson.Safe.t

(** Convert an ASL type to JSON *)
val ty_to_json : Asllib.AST.ty -> Yojson.Safe.t

(** Convert an ASL declaration to JSON *)
val decl_to_json : Asllib.AST.decl -> Yojson.Safe.t

(** Convert a complete ASL AST to JSON *)
val ast_to_json : Asllib.AST.t -> Yojson.Safe.t

(** Parse an ASL file and convert to JSON *)
val parse_file_to_json : ?with_stdlib:bool -> string -> Yojson.Safe.t

(** Parse an ASL string and convert to JSON *)
val parse_string_to_json : ?with_stdlib:bool -> string -> Yojson.Safe.t
