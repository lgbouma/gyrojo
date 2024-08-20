"""
Contents:

LaTeX value file I/O:
    update_latex_key_value_pair
    read_latex_key_value_pairs
    int_to_string

LaTeX table formatting:
    format_lowerlimit
    cast_to_int_string
    replace_nan_string
    format_prot_err
"""
import os, re
from os.path import join
from gyrojo.paths import PAPERDIR
from pathlib import Path
import numpy as np, pandas as pd

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

def read_latex_key_value_pairs(
    latexpath=join(PAPERDIR, "vals_manuscript.tex")
) -> dict:
    """
    Args:
        latexpath (str): The path to the LaTeX file containing the key-value pairs.

    Returns:
        dict: A dictionary containing the key-value pairs read from the file.
    """

    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }

    key_value_dict = {}

    with open(latexpath, "r") as file:
        for line in file:
            match = re.match(r"\\newcommand{\\(\w+)}{(.+)}", line.strip())
            if match:
                key = match.group(1)
                value = match.group(2)

                # Remove comma-padding from the value
                value = value.replace("{,}", "")

                # Check if the value contains a decimal point
                if value.lower() in number_words:
                    value = number_words[value.lower()]
                elif "." in value:
                    value = float(value)
                elif "$" in value:
                    value = str(value)
                else:
                    value = int(value)

                key_value_dict[key] = value

    return key_value_dict


def format_lowerlimit(value):
    if pd.isna(value):
        return np.nan
    else:
        return f"$> {int(value)}$"

def cast_to_int_string(value):
    if pd.isna(value):
        return np.nan
    else:
        if isinstance(value, float):
            return str(int(np.round(value,0)))
        elif isinstance(value, int):
            return str(int(value))
        elif isinstance(value, str):
            return value
        else:
            print(f'got value type {type(value)}')
            import IPython; IPython.embed()
            raise NotImplementedError

def replace_nan_string(value):
    if pd.isna(value) or 'nan' in str(value).lower():
        return '--'
    else:
        return value

def format_prot_err(row):
    if row['Prot'] <= 5:
        return f"{row['Prot_err']:.3f}"
    else:
        return f"{row['Prot_err']:.2f}"


def int_to_string(n):
    if n <= 9:
        numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        return numbers[n]
    else:
        return n
