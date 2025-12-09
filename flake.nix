{
  description = "ASL-MLIR";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    typix = {
      url = "github:loqusion/typix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      ...
    }:
    let
      overlay = import ./nix/overlay.nix;
    in
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      # Add supported platform here
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      flake = {
        overlays = rec {
          default = overlay;
        };
      };

      imports = [
        inputs.treefmt-nix.flakeModule
      ];

      perSystem =
        { system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              overlay
            ];
          };

          typixLib = inputs.typix.lib.${system};

          # Use doc building utilities from overlay
          docsLib = pkgs.callPackage ./nix/pkgs/asl-docs.nix {
            inherit typixLib;
            docSrc = ./doc;
          };
        in
        {
          _module.args.pkgs = pkgs;

          legacyPackages = pkgs;

          packages = {
            # Individual doc builds
            doc-GMPRational = docsLib.docs.GMPRational;
            doc-IR = docsLib.docs.IR;
            doc-Pass = docsLib.docs.Pass;
            doc-Rational = docsLib.docs.Rational;
            doc-Development = docsLib.docs.Development;

            # All docs combined
            docs = docsLib.all;
          };

          apps = {
            watch-docs = {
              type = "app";
              program = "${docsLib.watch}/bin/typst-watch";
            };
          };

          devShells = {
            default = pkgs.mkShell {
              buildInputs = with pkgs; [
                herdtools7
                typst
                asl-mlir
                ocaml
                cmake
                ninja
                gmp
              ];
            };
          };

          treefmt = {
            projectRootFile = "flake.nix";
            settings.on-unmatched = "debug";
            programs = {
              nixfmt.enable = true;
              scalafmt.enable = true;
              clang-format.enable = true;
            };
            settings.formatter = {
              nixfmt.excludes = [
                "*/generated.nix"
              ];
              scalafmt.includes = [
              ];
              clang-format.includes = [
                "mlir/**/*.cpp"
                "mlir/**/*.h"
                "mlir/**/*.hpp"
                "mlir/**/*.c"
                "mlir/**/*.cc"
                "mlir/**/*.td"
              ];
            };
          };
        };
    };
}
