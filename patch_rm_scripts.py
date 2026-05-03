"""Patch score_skywork_v2.py and score_armorm.py to handle BatchEncoding return from apply_chat_template."""
import re

for path in ["score_skywork_v2.py", "score_armorm.py"]:
    src = open(path).read()
    # Match either tokenize=True or no tokenize arg — both produce the same broken behavior
    pattern = re.compile(
        r'    ids = tok\.apply_chat_template\(convo(?:, tokenize=True)?, return_tensors="pt"\)\.to\(device\)\n'
        r'    with torch\.no_grad\(\):\n'
        r'        out = rm\(ids\)\n'
    )
    new_block = (
        '    text = tok.apply_chat_template(convo, tokenize=False)\n'
        '    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)\n'
        '    with torch.no_grad():\n'
        '        out = rm(**inputs)\n'
    )
    if pattern.search(src):
        src = pattern.sub(new_block, src)
        open(path, "w").write(src)
        print(f"patched {path}")
    else:
        print(f"NO MATCH in {path}")

print("---verification---")
for path in ["score_skywork_v2.py", "score_armorm.py"]:
    print(f"\n--- {path} ---")
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if "apply_chat_template" in line or "out = rm" in line or "inputs = tok" in line:
                print(f"  L{i}: {line.rstrip()}")
