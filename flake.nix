{
  description = "Simulations and inverse problems";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    custom-nixpkgs.url = "github:lukebodmer/custom_nixpkgs";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, custom-nixpkgs, pyproject-nix, ... }:
      let
        system = "x86_64-linux";

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ custom-nixpkgs.overlays.default ];
        };

	project = pyproject-nix.lib.project.loadPyproject {
          projectRoot = ./.;
	};

	python = pkgs.python312;

      in
	{
          devShells.${system}.default =
            let
              # Returns a function that can be passed to `python.withPackages`
              arg = project.renderers.withPackages { inherit python; };

              # Returns a wrapped environment (virtualenv like) with all our packages
              pythonEnv = python.withPackages arg;

            in
              # Create a devShell like normal.
              pkgs.mkShell {
		packages = [ pythonEnv ];

		shellHook = ''
		  export VIRTUAL_ENV="Wave Map"
		'';
	      };

	  # Build our package using `buildPythonPackage
	  packages.${system}.default =
            let
              # Returns an attribute set that can be passed to `buildPythonPackage`.
              attrs = project.renderers.buildPythonPackage { inherit python; };
            in
              # Pass attributes to buildPythonPackage.
              # Here is a good spot to add on any missing or custom attributes.
              python.pkgs.buildPythonPackage (attrs // { env.CUSTOM_ENVVAR = "hello"; });
    };
}

