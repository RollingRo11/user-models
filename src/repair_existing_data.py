import sys
from pathlib import Path
from typing import List

# Ensure local imports work when invoked as a module
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_synthetic_data import fallback_conversation_for


DATA_DIR = Path("data")


def count_utterances(content: str) -> int:
    return sum(
        1
        for ln in (content or "").splitlines()
        if ln.startswith("HUMAN:") or ln.startswith("ASSISTANT:")
    )


def repair_file(path: Path, force: bool = False) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        content = ""
    n = count_utterances(content)
    if n >= 6 and not force:
        return False

    # Infer label group and label from path: data/<group>/<label>/conv_xxxx.txt
    try:
        label = path.parent.name
        label_group = path.parent.parent.name
    except Exception:
        return False

    # Seed variation based on file path for diversity and stability
    seed = f"{label_group}:{label}:{path.stem}"
    fixed = fallback_conversation_for(label_group, label, seed=seed)
    path.write_text(fixed, encoding="utf-8")
    return True


def main(argv: List[str]) -> int:
    base = DATA_DIR
    force = False
    args = [a for a in argv[1:] if a]
    for a in list(args):
        if a in {"-f", "--force"}:
            force = True
            args.remove(a)
    if args:
        base = Path(args[0])
    if not base.exists():
        print(f"No such directory: {base}")
        return 1

    repaired = 0
    total = 0
    for txt in base.rglob("*.txt"):
        total += 1
        if repair_file(txt, force=force):
            repaired += 1
            print(f"Repaired: {txt}")

    print(f"Checked {total} files; repaired {repaired}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
