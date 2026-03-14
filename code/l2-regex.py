import re

pattern_name = r'[A-Za-z_][A-Za-z0-9_]*'
pattern_number = r'\d+(?:\.\d+)?'

examples = ['ABC', 'aBC', 'x = 123', 'var1 = 3.14']

for text in examples:
    names = re.findall(pattern_name, text)
    numbers = re.findall(pattern_number, text)

    print(f"Text: {text!r}")
    print(f"Names: {names}")
    print(f"Numbers: {numbers}")
    print("-" * 40)
