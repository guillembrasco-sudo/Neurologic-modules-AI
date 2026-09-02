import os
import json
from common import Tokens
import re
from pathlib import Path

PATH_FOLDER_NEW=r"D:\\Courses"
folders=["Kaggle - Computer vision", "Kaggle - Introduction to deep learning", "Kaggle - machine Learning"]
tokens = Tokens()

def cargar_file_ipynb(ruta_ipynb):
    """Load an .ipynb and return a list of tokenized markdown cell strings."""
    with open(ruta_ipynb, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _tokenize_markdown(source_lines, tokens: Tokens):
        out_lines = []
        in_code_block = False
        for raw in source_lines:
            line = raw.rstrip("\n")
            # detect code fence
            if line.startswith("```"):
                if in_code_block:
                    out_lines.append(tokens.code_end_token)
                    in_code_block = False
                else:
                    lang = line[3:].strip()
                    attrs = f'language="{lang}"' if lang else ""
                    out_lines.append(tokens.code_start_token + attrs + ">")
                    in_code_block = True
                continue

            # images: ![alt](url)
            img_match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if img_match:
                src = img_match.group(1)
                # replace the markdown image with img tokens
                img_tokenized = f"{tokens.img_start_token}src=\"{src}\">{tokens.img_end_token}"
                line = re.sub(r"!\[[^\]]*\]\([^)]+\)", img_tokenized, line)

            # links: [text](url)
            def _link_repl(m):
                text = m.group(1)
                href = m.group(2)
                return f"{tokens.link_start_token}href=\"{href}\">{text}{tokens.link_end_token}"

            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link_repl, line)

            # prefix each line with a line token to preserve boundaries
            out_lines.append(tokens.line_token + line if line else tokens.line_token)

        # join and wrap with start/end
        body = "".join(out_lines)
        return tokens.start_token + body + tokens.end_token

    # detect notebook-level language (kernelspec) fallback
    kernel_lang = data.get("metadata", {}).get("kernelspec", {}).get("language", "") or "python"

    tokenized_cells = []
    for cell in data.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            tokenized_cells.append(_tokenize_markdown(cell.get("source", []), tokens))
        elif ctype == "code":
            # determine language: per-cell metadata -> notebook kernelspec -> default
            lang = cell.get("metadata", {}).get("language", "") or kernel_lang or "python"
            attrs = f'language="{lang}"'
            # join code lines preserving formatting, prefix with line tokens to keep boundaries
            code_lines = []
            for raw in cell.get("source", []):
                code_lines.append(tokens.line_token + raw.rstrip("\n"))
            body = "".join(code_lines)
            tokenized = tokens.start_token + tokens.code_start_token + attrs + ">" + body + tokens.code_end_token + tokens.end_token
            tokenized_cells.append(tokenized)
        else:
            # unknown cell types are preserved as plain text
            body = "".join(tokens.line_token + l.rstrip("\n") for l in cell.get("source", []))
            tokenized_cells.append(tokens.start_token + body + tokens.end_token)

    return tokenized_cells

to_write = ""
print(Path.cwd().resolve())

for fold in folders:
    ruta = os.path.join(PATH_FOLDER_NEW, fold)
    for archivo in os.listdir(ruta):
            ruta_completa = os.path.join(ruta, archivo)

            if archivo.lower().endswith(".ipynb"):
                to_write += "\n".join(cargar_file_ipynb(ruta_completa)) + "\n"

with open(os.path.join(Path.cwd().resolve(), r"DATA\\data_courses.txt"), "wb") as f:
    f.write(to_write.encode("utf-8"))
