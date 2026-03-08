import sys
import ast
import dis
import tokenize
from io import BytesIO

def usage():
    print("Usage: python pycview.py <source.py> -ast|-ir|-tok")
    sys.exit(1)

def load_source(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def dump_ast(source, filename):
    tree = ast.parse(source, filename=filename)
    print(ast.dump(tree, indent=2))

def dump_ir(source, filename):
    code = compile(source, filename, "exec")
    dis.dis(code)

def dump_tokens(source):
    """
    Dump lexical tokens produced by Python tokenizer,
    including INDENT / DEDENT / NEWLINE / ENDMARKER.
    """
    reader = BytesIO(source.encode("utf-8")).readline
    for tok in tokenize.tokenize(reader):
        print(tok)

def main():
    if len(sys.argv) != 3:
        usage()

    filename = sys.argv[1]
    option = sys.argv[2]

    if option not in ("-ast", "-ir", "-tok"):
        usage()

    source = load_source(filename)

    if option == "-ast":
        dump_ast(source, filename)
    elif option == "-ir":
        dump_ir(source, filename)
    elif option == "-tok":
        dump_tokens(source)

if __name__ == "__main__":
    main()

