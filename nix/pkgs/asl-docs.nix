{
  lib,
  linkFarm,
  typixLib,
  docSrc,
}:

let
  src = typixLib.cleanTypstSource docSrc;

  # Typst packages from the official registry (including transitive deps)
  unstable_typstPackages = [
    {
      name = "fletcher";
      version = "0.5.3";
      hash = "sha256-fgiCsvIUhiXu4wYJjwXhGfcGoB60Asg4dCypnDe3AYc=";
    }
    # Required by fletcher
    {
      name = "cetz";
      version = "0.3.1";
      hash = "sha256-xCY+RLTciuq0LNcvIMoO3Lqm61YaziLgWGvL6uMZD58=";
    }
    # Required by cetz
    {
      name = "oxifmt";
      version = "0.2.0";
      hash = "sha256-JYVVcyze+qTzdb6wMG0Sm1rJ6TMqrogrx9GAZPD1Cug=";
    }
  ];

  # Common args for all doc builds
  commonArgs = {
    inherit src unstable_typstPackages;
  };

  # Build a single document (typix outputs a single PDF file)
  buildDoc =
    name:
    typixLib.buildTypstProject (
      commonArgs
      // {
        typstSource = "${name}.typ";
      }
    );

  # All documentation files
  docNames = [
    "GMPRational"
    "IR"
    "Pass"
    "Rational"
    "Development"
  ];

  # Individual doc derivations
  docs = lib.genAttrs docNames buildDoc;

  # All docs combined using linkFarm (since each doc is a single PDF file)
  all = linkFarm "asl-docs" (
    map (name: {
      name = "${name}.pdf";
      path = docs.${name};
    }) docNames
  );

  # Watch script for development
  watch = typixLib.watchTypstProject (
    commonArgs
    // {
      typstSource = "GMPRational.typ"; # Default to main doc
    }
  );

in
{
  inherit
    docs
    all
    watch
    unstable_typstPackages
    ;
}
