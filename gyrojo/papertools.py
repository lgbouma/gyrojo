import os, re
from os.path import join
from gyrojo.paths import PAPERDIR
from pathlib import Path

def update_latex_key_value_pair(
    latexkeyname, latexvalue,
    latexpath=join(PAPERDIR, "vals_manuscript.tex")
):
    """
    Add lines like "\newcommand{\nstarswithaliens}{1}" to a value file
    included in the manuscript.  This function searches through a
    LaTeX file for a given key-value pair. If the pair is found, it
    updates the value. If the pair is not found, it adds a new line
    with the key-value pair. The function prints the updates made to
    the file to stdout.

    If latexvalue is a long number (greater than 999), it will be
    comma-padded.  For example, 53421 will become "53{,}421".

    Args:
        latexkeyname (str): The name of the LaTeX key to update or add.
            Must contain only lowercase letters.
        latexvalue (str): The value to assign to the LaTeX key.
        latexpath (str, optional): The path to the LaTeX file.
            Defaults to "vals_stars.tex".

    Returns:
        None
    """

    if not os.path.exists(latexpath):
        Path(latexpath).touch()

    if not re.match(r'^[a-z]+$', latexkeyname):
        raise ValueError("latexkeyname must contain only lowercase letters.")

    if not isinstance(latexvalue, str):
        latexvalue = str(latexvalue)

    # Comma-pad long numbers in latexvalue
    if latexvalue.isdigit() and len(latexvalue) > 3:
        latexvalue = f"{latexvalue[:-3]}{{,}}{latexvalue[-3:]}"

    updated = False
    lines = []

    with open(latexpath, "r") as file:
        for line in file:
            if line.startswith(f"\\newcommand{{\\{latexkeyname}}}"):
                lines.append(f"\\newcommand{{\\{latexkeyname}}}{{{latexvalue}}}\n")
                updated = True
                print(f"Updated: \\newcommand{{\\{latexkeyname}}}{{{latexvalue}}}")
            else:
                lines.append(line)

    if not updated:
        lines.append(f"\\newcommand{{\\{latexkeyname}}}{{{latexvalue}}}\n")
        print(f"Added: \\newcommand{{\\{latexkeyname}}}{{{latexvalue}}}")

    with open(latexpath, "w") as file:
        file.writelines(lines)