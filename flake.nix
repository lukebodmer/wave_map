  {
  description = "Simulations and inverse problems";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    custom-nixpkgs.url = "github:lukebodmer/custom_nixpkgs";
  };

  outputs = { self, nixpkgs, custom-nixpkgs, ... }:
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ custom-nixpkgs.overlays.default ];
        };
      in
        {
          devShells.${system}.default = pkgs.mkShell {
            name = "default";
               
            packages = [
            # General packages
              # pkgs.hello-nix
              # pkgs.petsc
              # pkgs.mpich
              # pkgs.clangd
              #  # Python packages
              (pkgs.python312.withPackages (python-pkgs: [
              #  # packages for formatting/ IDE
              #  python-pkgs.pip
                python-pkgs.python-lsp-server
              #  # packages for code
                python-pkgs.gmsh
                python-pkgs.matplotlib
                python-pkgs.numpy
                python-pkgs.panel
		python-pkgs.pyvista
                python-pkgs.scipy
		python-pkgs.tomli
		python-pkgs.toml
              ]))
            ];

            # PETSC_DIR = "${pkgs.petsc}";

            shellHook = ''
              export VIRTUAL_ENV="Wave Map"
            '';
          };
        };
}

