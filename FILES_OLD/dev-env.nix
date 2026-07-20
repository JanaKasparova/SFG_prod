# Source - https://stackoverflow.com/a/79879457
# Posted by Zorgosto, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-10, License - CC BY-SA 4.0

{ pkgs ? import  (fetchTarball "https://github.com/nixos/nixpkgs/archive/nixos-25.11.tar.gz") {} }:

let
  pythonWithPkgs = pkgs.python312.withPackages (ps: with ps; [
    pip
    # other packages
  ]);
in
pkgs.mkShell {
  # Specify the packages needed in the development environment
  buildInputs = with pkgs; [
    pythonWithPkgs
    jetbrains.pycharm
  ];

  # Shell initialization script
  shellHook = ''
    export VENV_DIR="$PWD/.venv"

    # create the virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
      ${pythonWithPkgs}/bin/python -m venv $VENV_DIR
    fi

    # Activate the virtual environment
    source $VENV_DIR/bin/activate

    # Set up python paths
    export PYTHONPATH="${pythonWithPkgs}/lib/python3.12/site-packages:$PYTHONPATH"
    #export TCL_LIBRARY="${pkgs.tcl}/lib/tcl${pkgs.tcl.version}" # for tcl
    #export TK_LIBRARY="${pkgs.tk}/lib/tk${pkgs.tk.version}" # for tkinter    

    # Install dependencies if requirements.txt exists
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi


    # Display Python version for verification
    PYTHON_VERSION=$(python --version)
    echo ""
    echo -e "\033[1;32mWelcome to your development environment!\033[0m Running: (\033[1;34m$PYTHON_VERSION\033[0m)"
    '';
}

