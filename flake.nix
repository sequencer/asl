{
  description = "ASL-MLIR";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
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
        in
        {
          _module.args.pkgs = pkgs;

          legacyPackages = pkgs;

          devShells = {
            default = pkgs.mkShell {
              buildInputs = with pkgs; [
                herdtools7
                typst
                asl-mlir
                ocaml
                cmake
                ninja
              ];
              shellHook = ''
                echo "Building MLIR project in mlir/build"
                mkdir -p mlir/build
                (cd mlir/build && cmake -G Ninja .. && ninja)

                echo "Copying herdtools7 package to reference/herdtools7"
                mkdir -p reference/herdtools7
                cp -rL --no-preserve=ownership ${pkgs.herdtools7}/* reference/herdtools7/
                export OCAMLPATH=${pkgs.herdtools7}/lib/:$OCAMLPATH
              '';
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
              ];
            };
          };
        };
    };
}
