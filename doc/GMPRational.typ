// GMP Dialect Implementation Plan
// Based on OCaml Zarith Library API Design
// Reference: https://github.com/ocaml/Zarith
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set document(title: "GMP Dialect Implementation Plan", author: "ASL-MLIR Team")
#set page(numbering: "1", margin: (x: 1.5cm, y: 2cm))
#set heading(numbering: "1.1")
#set text(font: "New Computer Modern", size: 11pt)

#align(center)[
  #text(size: 24pt, weight: "bold")[GMP Dialect Implementation Plan]

  #v(0.5em)
  #text(size: 14pt)[ASL-MLIR Arbitrary Precision Arithmetic]

  #v(0.3em)
  #text(size: 11pt, style: "italic")[Based on OCaml Zarith Library API Design]

  #v(1em)
]

#outline(indent: auto)

#pagebreak()

= Overview

This document defines the GMP dialect for ASL-MLIR, providing arbitrary-precision integer ($ZZ$) and rational ($QQ$) arithmetic. The API design follows the OCaml Zarith library conventions.

== Design Principles

Following Zarith's design:

+ *Clean separation*: `Z` module for integers, `Q` module for rationals
+ *Canonical forms*: Rationals always reduced with positive denominators
+ *Multiple division modes*: Truncated, floor, ceiling, Euclidean
+ *Rich number theory*: GCD, LCM, primality, modular arithmetic
+ *Compile-time evaluation*: Constant folding without runtime dependency

== Types

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, left, center, left),
  [*MLIR Type*], [*Zarith*], [*Symbol*], [*Description*],
  [`!gmp.z`], [`Z.t`], [$ZZ$], [Arbitrary precision integer],
  [`!gmp.q`], [`Q.t`], [$QQ$], [Arbitrary precision rational (canonical form)],
)

= File Structure

```
asl-mlir/
├── include/GMP/
│   ├── CMakeLists.txt       # TableGen generation
│   ├── GMP.td               # Main include
│   ├── GMPDialect.td        # Dialect definition
│   ├── GMPTypes.td          # Z and Q types
│   ├── GMPOps.td            # All operations
│   └── GMPAttributes.td     # Constant attributes
├── lib/GMP/IR/
│   ├── CMakeLists.txt
│   ├── GMPDialect.cpp
│   ├── GMPTypes.cpp
│   ├── GMPOps.cpp           # With fold() implementations
│   └── GMPAttributes.cpp
└── test/GMP/
    ├── z-ops.mlir           # Integer operation tests
    ├── q-ops.mlir           # Rational operation tests
    ├── constant-fold.mlir   # Folding tests
    └── invalid.mlir         # Error tests
```

= Z Module: Arbitrary Precision Integers ($ZZ$)

The `Z` module provides operations on arbitrary-precision integers, following Zarith's `Z` module API.

== Construction

Constants use MLIR attributes (backed by APInt), runtime values use conversion ops:

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith Equivalent*], [*Description*],
  [`gmp.z.constant <attr>`], [`Z.zero`, `Z.one`, etc.], [Constant from ZAttr (APInt-backed)],
  [`gmp.z.of_string "42"`], [`Z.of_string "42"`], [Parse from decimal string (attr only)],
  [`gmp.z.from_int %x`], [`Z.of_int64 x`], [Convert from MLIR integer (any width)],
)

Attributes:
- `#gmp.z<0>` - Zero constant
- `#gmp.z<1>` - One constant
- `#gmp.z<-1>` - Minus one constant
- `#gmp.z<12345678901234567890>` - Arbitrary precision literal

```mlir
// Constants via attributes (compile-time, no runtime cost)
%zero = gmp.z.constant #gmp.z<0> : !gmp.z
%one = gmp.z.constant #gmp.z<1> : !gmp.z
%big = gmp.z.constant #gmp.z<12345678901234567890> : !gmp.z

// Runtime conversion from MLIR integer types (uses APInt internally)
%m = gmp.z.from_int %x : i64 -> !gmp.z
%n = gmp.z.from_int %y : i128 -> !gmp.z   // Any integer width supported
```

== Basic Arithmetic

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  align: (left, left, left, left),
  [*Operation*], [*Zarith*], [*Math*], [*Description*],
  [`gmp.z.succ %a`], [`Z.succ`], [$a + 1$], [Successor],
  [`gmp.z.pred %a`], [`Z.pred`], [$a - 1$], [Predecessor],
  [`gmp.z.abs %a`], [`Z.abs`], [$|a|$], [Absolute value],
  [`gmp.z.neg %a`], [`Z.neg`], [$-a$], [Negation],
  [`gmp.z.add %a, %b`], [`Z.add`], [$a + b$], [Addition],
  [`gmp.z.sub %a, %b`], [`Z.sub`], [$a - b$], [Subtraction],
  [`gmp.z.mul %a, %b`], [`Z.mul`], [$a times b$], [Multiplication],
)

```mlir
%sum = gmp.z.add %a, %b : !gmp.z
%prod = gmp.z.mul %a, %b : !gmp.z
%next = gmp.z.succ %n : !gmp.z
```

== Division Operations

Zarith provides multiple division modes. The key difference is how negative numbers are handled:

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Mode*], [*Quotient rounds toward*], [*Remainder sign*],
  [Truncated (`div`, `rem`)], [Zero], [Same as dividend],
  [Floor (`fdiv`, `frem`)], [$-infinity$], [Same as divisor],
  [Ceiling (`cdiv`, `crem`)], [$+infinity$], [Opposite of divisor],
  [Euclidean (`ediv`, `erem`)], [$-infinity$ for positive divisor], [Always non-negative],
)

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.div %a, %b`], [`Z.div`], [Truncated quotient (toward zero)],
  [`gmp.z.rem %a, %b`], [`Z.rem`], [Truncated remainder],
  [`gmp.z.div_rem %a, %b`], [`Z.div_rem`], [Both quotient and remainder],
  [`gmp.z.fdiv %a, %b`], [`Z.fdiv`], [Floor quotient (toward $-infinity$)],
  [`gmp.z.cdiv %a, %b`], [`Z.cdiv`], [Ceiling quotient (toward $+infinity$)],
  [`gmp.z.ediv %a, %b`], [`Z.ediv`], [Euclidean quotient],
  [`gmp.z.erem %a, %b`], [`Z.erem`], [Euclidean remainder (always $gt.eq 0$)],
  [`gmp.z.divexact %a, %b`], [`Z.divexact`], [Exact division (when $b | a$)],
)

```mlir
// Truncated division (like C)
%q = gmp.z.div %a, %b : !gmp.z      // -7 / 3 = -2
%r = gmp.z.rem %a, %b : !gmp.z      // -7 rem 3 = -1

// Floor division (like Python)
%q = gmp.z.fdiv %a, %b : !gmp.z     // -7 fdiv 3 = -3
```

== Divisibility Tests

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.divisible %a, %b`], [`Z.divisible`], [Tests if $b | a$],
  [`gmp.z.congruent %a, %b, %c`], [`Z.congruent`], [Tests if $a equiv b (mod c)$],
)

== Bitwise Operations

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.logand %a, %b`], [`Z.logand`], [Bitwise AND],
  [`gmp.z.logor %a, %b`], [`Z.logor`], [Bitwise OR],
  [`gmp.z.logxor %a, %b`], [`Z.logxor`], [Bitwise XOR],
  [`gmp.z.lognot %a`], [`Z.lognot`], [Bitwise NOT (one's complement)],
  [`gmp.z.shift_left %a, %n`], [`Z.shift_left`], [Left shift by $n$ bits],
  [`gmp.z.shift_right %a, %n`], [`Z.shift_right`], [Arithmetic right shift],
  [`gmp.z.shift_right_trunc %a, %n`], [`Z.shift_right_trunc`], [Truncating right shift],
  [`gmp.z.testbit %a, %n`], [`Z.testbit`], [Test bit at position $n$],
  [`gmp.z.popcount %a`], [`Z.popcount`], [Count 1-bits],
  [`gmp.z.hamdist %a, %b`], [`Z.hamdist`], [Hamming distance],
  [`gmp.z.numbits %a`], [`Z.numbits`], [Number of significant bits],
  [`gmp.z.trailing_zeros %a`], [`Z.trailing_zeros`], [Count trailing zeros],
)

```mlir
%and = gmp.z.logand %a, %b : !gmp.z
%shl = gmp.z.shift_left %a, %n : (!gmp.z, i32) -> !gmp.z
%bit = gmp.z.testbit %a, %pos : (!gmp.z, i32) -> i1
```

== Bit Extraction

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.extract %a, %off, %len`], [`Z.extract`], [Extract $"len"$ bits starting at $"off"$ (unsigned)],
  [`gmp.z.signed_extract %a, %off, %len`], [`Z.signed_extract`], [Extract with sign extension],
)

== Comparison and Ordering

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.compare %a, %b`], [`Z.compare`], [Returns $-1$, $0$, or $1$],
  [`gmp.z.equal %a, %b`], [`Z.equal`], [$a = b$],
  [`gmp.z.leq %a, %b`], [`Z.leq`], [$a lt.eq b$],
  [`gmp.z.geq %a, %b`], [`Z.geq`], [$a gt.eq b$],
  [`gmp.z.lt %a, %b`], [`Z.lt`], [$a < b$],
  [`gmp.z.gt %a, %b`], [`Z.gt`], [$a > b$],
  [`gmp.z.sign %a`], [`Z.sign`], [Sign: $-1$, $0$, or $1$],
  [`gmp.z.min %a, %b`], [`Z.min`], [Minimum],
  [`gmp.z.max %a, %b`], [`Z.max`], [Maximum],
)

== Parity Tests

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.is_even %a`], [`Z.is_even`], [Tests if $a$ is even],
  [`gmp.z.is_odd %a`], [`Z.is_odd`], [Tests if $a$ is odd],
)

== Powers and Roots

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.pow %a, %n`], [`Z.pow`], [$a^n$],
  [`gmp.z.sqrt %a`], [`Z.sqrt`], [$floor(sqrt(a))$],
  [`gmp.z.sqrt_rem %a`], [`Z.sqrt_rem`], [$(floor(sqrt(a)), a - floor(sqrt(a))^2)$],
  [`gmp.z.root %a, %n`], [`Z.root`], [$floor(root(n, a))$],
  [`gmp.z.rootrem %a, %n`], [`Z.rootrem`], [Root with remainder],
  [`gmp.z.perfect_power %a`], [`Z.perfect_power`], [Is $a = b^k$ for some $k > 1$?],
  [`gmp.z.perfect_square %a`], [`Z.perfect_square`], [Is $a$ a perfect square?],
  [`gmp.z.log2 %a`], [`Z.log2`], [$floor(log_2(a))$],
  [`gmp.z.log2up %a`], [`Z.log2up`], [$ceil(log_2(a))$],
)

== Number Theory

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.gcd %a, %b`], [`Z.gcd`], [$gcd(a, b)$],
  [`gmp.z.gcdext %a, %b`], [`Z.gcdext`], [Extended GCD: $(g, s, t)$ where $g = a s + b t$],
  [`gmp.z.lcm %a, %b`], [`Z.lcm`], [$"lcm"(a, b)$],
  [`gmp.z.powm %a, %b, %m`], [`Z.powm`], [$a^b mod m$],
  [`gmp.z.powm_sec %a, %b, %m`], [`Z.powm_sec`], [Constant-time modular exponentiation],
  [`gmp.z.invert %a, %m`], [`Z.invert`], [$a^(-1) mod m$],
  [`gmp.z.probab_prime %a, %reps`], [`Z.probab_prime`], [Primality test],
  [`gmp.z.nextprime %a`], [`Z.nextprime`], [Next prime $> a$],
  [`gmp.z.jacobi %a, %b`], [`Z.jacobi`], [Jacobi symbol],
  [`gmp.z.legendre %a, %b`], [`Z.legendre`], [Legendre symbol],
  [`gmp.z.kronecker %a, %b`], [`Z.kronecker`], [Kronecker symbol],
  [`gmp.z.remove %a, %b`], [`Z.remove`], [Remove factor: $(a / b^k, k)$ where $b^k | a$],
)

== Factorial and Combinatorics

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.fac %n`], [`Z.fac`], [$n!$],
  [`gmp.z.fac2 %n`], [`Z.fac2`], [$n!!$ (double factorial)],
  [`gmp.z.facM %n, %m`], [`Z.facM`], [$n!^((m))$ (m-th factorial)],
  [`gmp.z.primorial %n`], [`Z.primorial`], [Product of primes $lt.eq n$],
  [`gmp.z.bin %n, %k`], [`Z.bin`], [$binom(n, k)$],
  [`gmp.z.fib %n`], [`Z.fib`], [$F_n$ (Fibonacci)],
  [`gmp.z.lucnum %n`], [`Z.lucnum`], [$L_n$ (Lucas number)],
)

== Conversions

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.z.to_i32 %a`], [`Z.to_int32`], [Convert to i32 (may overflow)],
  [`gmp.z.to_i64 %a`], [`Z.to_int64`], [Convert to i64 (may overflow)],
  [`gmp.z.to_f64 %a`], [`Z.to_float`], [Convert to f64],
  [`gmp.z.to_string %a`], [`Z.to_string`], [Convert to decimal string],
  [`gmp.z.fits_i32 %a`], [`Z.fits_int32`], [Can convert without overflow?],
  [`gmp.z.fits_i64 %a`], [`Z.fits_int64`], [Can convert without overflow?],
)

= Q Module: Arbitrary Precision Rationals ($QQ$)

The `Q` module provides operations on arbitrary-precision rationals, following Zarith's `Q` module API. Rationals are represented as `{num: Z.t; den: Z.t}` in canonical form (reduced, positive denominator).

== Construction

Constants use MLIR attributes (backed by pair of APInt: num/den):

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.constant <attr>`], [`Q.zero`, `Q.one`, etc.], [Constant from QAttr (pair of APInt)],
  [`gmp.q.make %num, %den`], [`Q.make`], [Create from two `!gmp.z` values],
)

Attributes:
- `#gmp.q<0>` - Zero constant (0/1)
- `#gmp.q<1>` - One constant (1/1)
- `#gmp.q<-1>` - Minus one constant (-1/1)
- `#gmp.q<3/4>` - Rational literal (always canonical form)
- `#gmp.q<355/113>` - Pi approximation

```mlir
// Constants via attributes (compile-time, no runtime cost)
%zero = gmp.q.constant #gmp.q<0> : !gmp.q
%half = gmp.q.constant #gmp.q<1/2> : !gmp.q
%pi_approx = gmp.q.constant #gmp.q<355/113> : !gmp.q

// Runtime construction from gmp.z values
%r = gmp.q.make %num, %den : (!gmp.z, !gmp.z) -> !gmp.q
```

== Special Values

Zarith's Q module supports special values for mathematical completeness. These are represented as attributes:

- `#gmp.q<inf>` - $+infinity$ (1/0)
- `#gmp.q<-inf>` - $-infinity$ (-1/0)
- `#gmp.q<undef>` - Undefined (0/0)

```mlir
%inf = gmp.q.constant #gmp.q<inf> : !gmp.q
%neg_inf = gmp.q.constant #gmp.q<-inf> : !gmp.q
%undef = gmp.q.constant #gmp.q<undef> : !gmp.q
```

== Component Access

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.num %a`], [`Q.num`], [Get numerator as `!gmp.z`],
  [`gmp.q.den %a`], [`Q.den`], [Get denominator as `!gmp.z`],
)

```mlir
%n = gmp.q.num %a : !gmp.q -> !gmp.z
%d = gmp.q.den %a : !gmp.q -> !gmp.z
```

== Classification

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.classify %a`], [`Q.classify`], [Returns kind: ZERO, INF, MINF, UNDEF, NZERO],
  [`gmp.q.is_real %a`], [`Q.is_real`], [True if finite (not inf, not undef)],
)

Classification kinds (following Zarith):
- `ZERO`: The value is 0
- `INF`: Positive infinity ($+infinity$)
- `MINF`: Negative infinity ($-infinity$)
- `UNDEF`: Undefined (0/0)
- `NZERO`: A non-zero finite rational

== Comparison and Ordering

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.compare %a, %b`], [`Q.compare`], [Returns $-1$, $0$, or $1$],
  [`gmp.q.equal %a, %b`], [`Q.equal`], [$a = b$],
  [`gmp.q.leq %a, %b`], [`Q.leq`], [$a lt.eq b$],
  [`gmp.q.geq %a, %b`], [`Q.geq`], [$a gt.eq b$],
  [`gmp.q.lt %a, %b`], [`Q.lt`], [$a < b$],
  [`gmp.q.gt %a, %b`], [`Q.gt`], [$a > b$],
  [`gmp.q.sign %a`], [`Q.sign`], [Sign: $-1$, $0$, or $1$],
  [`gmp.q.min %a, %b`], [`Q.min`], [Minimum],
  [`gmp.q.max %a, %b`], [`Q.max`], [Maximum],
)

== Basic Arithmetic

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.neg %a`], [`Q.neg`], [$-a$],
  [`gmp.q.abs %a`], [`Q.abs`], [$|a|$],
  [`gmp.q.add %a, %b`], [`Q.add`], [$a + b$],
  [`gmp.q.sub %a, %b`], [`Q.sub`], [$a - b$],
  [`gmp.q.mul %a, %b`], [`Q.mul`], [$a times b$],
  [`gmp.q.div %a, %b`], [`Q.div`], [$a div b$],
  [`gmp.q.inv %a`], [`Q.inv`], [$1 / a$],
)

```mlir
%sum = gmp.q.add %a, %b : !gmp.q
%prod = gmp.q.mul %a, %b : !gmp.q
%recip = gmp.q.inv %a : !gmp.q
```

== Scaling by Powers of 2

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.mul_2exp %a, %n`], [`Q.mul_2exp`], [$a times 2^n$],
  [`gmp.q.div_2exp %a, %n`], [`Q.div_2exp`], [$a / 2^n$],
)

== Conversions

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Zarith*], [*Description*],
  [`gmp.q.to_bigint %a`], [`Q.to_bigint`], [Truncate to `!gmp.z`],
  [`gmp.q.to_i32 %a`], [`Q.to_int32`], [Truncate to i32],
  [`gmp.q.to_i64 %a`], [`Q.to_int64`], [Truncate to i64],
  [`gmp.q.to_f64 %a`], [`Q.to_float`], [Convert to f64],
  [`gmp.q.to_string %a`], [`Q.to_string`], [Convert to string "num/den"],
)

== Additional Q Operations (beyond Zarith)

For ASL compatibility, we add rounding operations:

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, left),
  [*Operation*], [*Description*], [*Example*],
  [`gmp.q.floor %a`], [$floor(a)$ toward $-infinity$], [$floor(7/3) = 2$, $floor(-7/3) = -3$],
  [`gmp.q.ceil %a`], [$ceil(a)$ toward $+infinity$], [$ceil(7/3) = 3$, $ceil(-7/3) = -2$],
  [`gmp.q.trunc %a`], [Truncate toward 0], [$"trunc"(7/3) = 2$, $"trunc"(-7/3) = -2$],
  [`gmp.q.round %a`], [Round to nearest (half away from 0)], [$"round"(5/2) = 3$],
)

= Canonicalization Patterns

Following Zarith's algebraic identities:

== Z Patterns

#table(
  columns: (auto, auto),
  inset: 8pt,
  [*Pattern*], [*Transformation*],
  [`z.add %x, z.zero`], [`%x`],
  [`z.sub %x, z.zero`], [`%x`],
  [`z.mul %x, z.one`], [`%x`],
  [`z.mul %x, z.zero`], [`z.zero`],
  [`z.div %x, z.one`], [`%x`],
  [`z.sub %x, %x`], [`z.zero`],
  [`z.neg (z.neg %x)`], [`%x`],
  [`z.abs (z.abs %x)`], [`z.abs %x`],
)

== Q Patterns

#table(
  columns: (auto, auto),
  inset: 8pt,
  [*Pattern*], [*Transformation*],
  [`q.add %x, q.zero`], [`%x`],
  [`q.mul %x, q.one`], [`%x`],
  [`q.mul %x, q.zero`], [`q.zero`],
  [`q.div %x, q.one`], [`%x`],
  [`q.mul %x, (q.inv %x)`], [`q.one` (if $x != 0$)],
  [`q.inv (q.inv %x)`], [`%x`],
  [`q.neg (q.neg %x)`], [`%x`],
)

= GMP Runtime Lowering

== Z Operations to GMP

#table(
  columns: (auto, auto),
  inset: 6pt,
  [*MLIR Operation*], [*GMP Function*],
  [`z.add`], [`mpz_add`],
  [`z.sub`], [`mpz_sub`],
  [`z.mul`], [`mpz_mul`],
  [`z.div`], [`mpz_tdiv_q`],
  [`z.rem`], [`mpz_tdiv_r`],
  [`z.fdiv`], [`mpz_fdiv_q`],
  [`z.cdiv`], [`mpz_cdiv_q`],
  [`z.ediv`], [`mpz_fdiv_q` (positive divisor) / `mpz_cdiv_q` (negative)],
  [`z.gcd`], [`mpz_gcd`],
  [`z.lcm`], [`mpz_lcm`],
  [`z.powm`], [`mpz_powm`],
  [`z.invert`], [`mpz_invert`],
  [`z.logand`], [`mpz_and`],
  [`z.logor`], [`mpz_ior`],
  [`z.logxor`], [`mpz_xor`],
  [`z.lognot`], [`mpz_com`],
)

== Q Operations to GMP

#table(
  columns: (auto, auto),
  inset: 6pt,
  [*MLIR Operation*], [*GMP Function*],
  [`q.make`], [`mpq_set_num` + `mpq_set_den` + `mpq_canonicalize`],
  [`q.add`], [`mpq_add`],
  [`q.sub`], [`mpq_sub`],
  [`q.mul`], [`mpq_mul`],
  [`q.div`], [`mpq_div`],
  [`q.inv`], [`mpq_inv`],
  [`q.neg`], [`mpq_neg`],
  [`q.abs`], [`mpq_abs`],
  [`q.num`], [`mpq_numref` + `mpz_set`],
  [`q.den`], [`mpq_denref` + `mpz_set`],
)

= Test Framework

The test infrastructure uses differential testing: GMP dialect operations are lowered to EmitC, compiled with clang against libgmp, and compared against reference implementations.

== Test Pipeline


#align(center)[
  #diagram(
    node-stroke: 0.5pt,
    node-inset: 8pt,
    spacing: 2em,

    node((0, 0), [GMP MLIR\ `.mlir`], name: <mlir>),
    node((1, 0), [EmitC\ `.c`], name: <emitc>),
    node((2, 0), [Clang\ `binary`], name: <clang>),
    node((3, 0), [Execute\ & Verify], name: <exec>),

    edge(<mlir>, <emitc>, "->", [`--convert-gmp-to-emitc`]),
    edge(<emitc>, <clang>, "->", [`-lgmp`]),
    edge(<clang>, <exec>, "->", [run]),

    node((3, 1), [Reference\ (Zarith/gmpy2)], name: <ref>),
    edge(<ref>, <exec>, "->", [diff]),
  )
]

== Directory Structure

```
test/GMP/
├── lit.cfg.py              # LIT configuration
├── z-arithmetic.mlir       # Z operation tests
├── z-division.mlir         # Division mode tests
├── z-bitwise.mlir          # Bitwise operation tests
├── z-number-theory.mlir    # GCD, primality, etc.
├── q-arithmetic.mlir       # Q operation tests
├── q-special.mlir          # inf, -inf, undef handling
├── constant-fold.mlir      # Compile-time folding tests
└── invalid.mlir            # Error/verification tests
```

== EmitC Lowering

Each GMP operation lowers to EmitC calls using libgmp:

```mlir
// Input: GMP dialect
%r = gmp.z.add %a, %b : !gmp.z

// Output: EmitC (after --convert-gmp-to-emitc)
%r = emitc.call_opaque "mpz_add" (%out, %a, %b) : (!emitc.ptr<!emitc.opaque<"mpz_t">>, ...) -> ()
```

== Test Structure

Each test file contains:
1. MLIR functions using GMP dialect
2. RUN lines for the test pipeline
3. CHECK lines for FileCheck verification

```mlir
// RUN: gmp-opt %s --convert-gmp-to-emitc | emitc-translate --mlir-to-c > %t.c
// RUN: clang %t.c -lgmp -o %t.exe
// RUN: %t.exe | FileCheck %s

func.func @test_z_add() {
  %a = gmp.z.constant #gmp.z<123456789012345678901234567890> : !gmp.z
  %b = gmp.z.constant #gmp.z<987654321098765432109876543210> : !gmp.z
  %r = gmp.z.add %a, %b : !gmp.z
  gmp.z.print %r  // CHECK: 1111111110111111111011111111100
  return
}
```

== Differential Testing

For correctness verification, compare against Python gmpy2 (GMP bindings):

```bash
# Generate test vectors
python3 scripts/gen_test_vectors.py > test_vectors.txt

# Run MLIR version
./build/bin/gmp-opt test.mlir --convert-gmp-to-emitc | \
  emitc-translate --mlir-to-c | clang -xc - -lgmp -o test && ./test

# Compare with Python gmpy2 reference
python3 -c "import gmpy2; print(gmpy2.mpz('123') + gmpy2.mpz('456'))"
```

== Edge Cases to Test

#table(
  columns: (auto, auto),
  inset: 6pt,
  [*Category*], [*Test Cases*],
  [Z Division], [Division by zero, negative divisors, all 4 modes],
  [Z Overflow], [Values exceeding 64-bit, 128-bit boundaries],
  [Q Canonical], [Reduction, sign normalization],
  [Q Special], [inf + inf, inf - inf, 0/0 propagation],
  [Q Division], [Division by zero → inf/-inf],
)

= Summary

This design follows Zarith's proven API patterns:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  [*Module*], [*Operations*], [*Key Features*],
  [Z], [50+], [Multiple division modes, number theory, bitwise],
  [Q], [25+], [Canonical form, special values, scaling],
)

Key design decisions:
- *Zarith naming*: `div`/`fdiv`/`cdiv`/`ediv` for division modes
- *Canonical forms*: Q always reduced with positive denominator
- *Compile-time folding*: LLVM APInt (no GMP build dependency)
- *Special values*: Q supports inf, minus_inf, undef
- *Component access*: Q.num, Q.den for direct access
