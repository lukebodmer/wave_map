{
  description = "A Python Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    custom-nixpkgs.url = "github:lukebodmer/custom_nixpkgs";
  };


  outputs = { self, nixpkgs, custom-nixpkgs, ... }:
    let
      system = "x86_64-linux";
      ## Import nixpkgs:

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ custom-nixpkgs.overlays.default ];
      };

      ## Read pyproject.toml file:
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);

      ## Get project specification:
      project = pyproject.project;

      ## Get the wave_map package:
      package = pkgs.python3Packages.buildPythonPackage {
        ## Set the package name:
        pname = project.name;

        ## Inherit the package version:
        inherit (project) version;

        ## Set the package format:
        format = "pyproject";

        ## Set the package source:
        src = ./.;

        ## Specify the build system to use:
        build-system = with pkgs.python3Packages; [
          setuptools
        ];
        ## Specify production dependencies:
        propagatedBuildInputs = [
          pkgs.python3.pkgs.cppimport
          pkgs.python3Packages.distutils
          pkgs.python3Packages.gmsh
          pkgs.python3Packages.numpy
          pkgs.python3Packages.pyvista
          pkgs.python3Packages.panel
          pkgs.python3Packages.scipy
          pkgs.python3Packages.sklearn-compat
          pkgs.python3Packages.tomli
          pkgs.python3Packages.toml
        ];

      };

      ## Make our package editable:
      editablePackage = pkgs.python3.pkgs.mkPythonEditablePackage {
        pname = project.name;
        inherit (project) scripts version;
        root = "$PWD/src";
      };
    in
      {
        ## Project packages output:
        packages = {
          "${project.name}" = package;
          default = self.packages.${system}.${project.name};
        };

        ## Project development shell output:
        devShells.${system}.default =
          pkgs.mkShell {
            inputsFrom = [
              package
            ];

            buildInputs = [
              # my package
              editablePackage

	      # tools
              pkgs.python3Packages.python-lsp-server
            ];

          };
      };
}
