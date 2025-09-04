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
	config.allowUnfree = true;
      };

      ## Choose a specific Python version for all packages
      python = pkgs.python312;
      pythonPackages = python.pkgs;

      ## Read pyproject.toml file:
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);

      ## Get project specification:
      project = pyproject.project;

      ## Get the wave_map package:
      package = pythonPackages.buildPythonPackage {
        ## Set the package name:
        pname = project.name;

        ## Inherit the package version:
        inherit (project) version;

        ## Set the package format:
        format = "pyproject";

        ## Set the package source:
        src = ./.;

        ## Specify the build system to use:
        build-system = with pythonPackages; [
          setuptools
        ];

        ## Specify production dependencies:
        propagatedBuildInputs = [
          pythonPackages.cppimport # this one is in custom-nixpkgs
          pythonPackages.distutils
          pythonPackages.gmsh
          #pythonPackages.numpy
          pythonPackages.cupy
          pythonPackages.pyvista
          pythonPackages.panel
          pythonPackages.scipy
          pythonPackages.sklearn-compat
          pythonPackages.tomli
          pythonPackages.toml
        ];

      };

      ## Make our package editable:
      editablePackage = pythonPackages.mkPythonEditablePackage {
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

	      pkgs.cudatoolkit

              # tools
              pythonPackages.python-lsp-server
              pythonPackages.flake8
            ];

            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH"
              echo "Entering Python development shell with CuPy."
            '';
          };
      };
}
