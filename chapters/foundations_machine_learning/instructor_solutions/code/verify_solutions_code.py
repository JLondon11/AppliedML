from pathlib import Path
import py_compile

root = Path(__file__).resolve().parent
files = sorted(p for p in root.glob('*.py') if p.name != Path(__file__).name)
failed=[]
for p in files:
    try:
        py_compile.compile(str(p), doraise=True)
        print('PASS', p.name)
    except Exception as e:
        failed.append((p.name, str(e)))
        print('FAIL', p.name, e)
if failed:
    raise SystemExit(1)
print(f'Validated {len(files)} Python solution scripts.')
