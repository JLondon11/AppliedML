"""Fail-closed correspondence check of real Chapter 6 LaTeX figures and inventory.

This check never certifies visual placement, scientific data or PDF typesetting.
The current manuscript's figure environments must map 1:1 to exact source-linked
inventory records. Complex custom TeX that this static parser cannot inspect
must undergo a documented independent audit; it is never automatically passed.
"""
import argparse
import json
import re
from pathlib import Path

FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}", re.S)
TEX_INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")

def strip_comments(source):
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in source.splitlines())

def braced(source, command):
    """Read balanced command arguments (not full TeX expansion)."""
    pattern = re.compile(r"\\" + re.escape(command) + r"(?:\[[^\]]*\])?\s*\{")
    items=[]
    for match in pattern.finditer(source):
        start=match.end();depth=1;i=start
        while i<len(source) and depth:
            if source[i]=='{' and (i==0 or source[i-1]!='\\'):
                depth+=1
            elif source[i]=='}' and (i==0 or source[i-1]!='\\'):
                depth-=1
            i+=1
        if depth==0:
            items.append(source[start:i-1])
    return items

def tex_files(main, root):
    seen=set();stack=[main];texts=[];errors=[]
    while stack:
        current=stack.pop()
        current=current.resolve()
        if current in seen:continue
        seen.add(current)
        if not current.is_relative_to(root) or not current.is_file():
            errors.append(f"Unsafe or absent TeX include: {current}");continue
        data=strip_comments(current.read_text(encoding='utf-8'))
        texts.append((str(current.relative_to(root)),data))
        for name in TEX_INPUT.findall(data):
            sub=(current.parent / (name if name.endswith('.tex') else name+'.tex')).resolve()
            stack.append(sub)
    return texts,errors

def audit(root, inventory):
    root=Path(root).resolve()
    inventory=Path(inventory)
    if not inventory.is_file():
        return {'status':'BLOCKED','issues':['Missing current manuscript figure inventory'],'source_environments':0}
    try:
        inv=json.loads(inventory.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'status':'BLOCKED','issues':[f'Unreadable figure inventory: {exc}'],'source_environments':0}
    issues=[]
    path=inv.get('manuscript_tex')
    if not isinstance(path,str) or not path.endswith('.tex'):
        return {'status':'BLOCKED','issues':['Current main manuscript_tex missing or invalid'],'source_environments':0}
    main=(root/path).resolve()
    texts,errors=tex_files(main,root);issues.extend(errors)
    blocks=[]
    for source,text in texts:
        for match in FIGURE_ENV.finditer(text):
            body=match.group(1)
            labels=braced(body,'label');images=braced(body,'includegraphics');captions=braced(body,'caption')
            if len(labels)!=1 or len(images)!=1 or len(captions)!=1:
                issues.append(f"{source}: ambiguous or incomplete figure environment")
                continue
            blocks.append({'tex_file':source,'label':labels[0].strip(),
                           'image':images[0].strip(),'caption':' '.join(captions[0].split())})
    listed=inv.get('figures')
    if not isinstance(listed,list):
        return {'status':'BLOCKED','issues':issues+['Missing figures list'],'source_environments':len(blocks)}
    labels=[b['label'] for b in blocks]
    if len(labels)!=len(set(labels)):issues.append('Repeated LaTeX figure label')
    invlabels=[]
    for row in listed:
        if not isinstance(row,dict):
            issues.append('Non-object inventory figure');continue
        label=row.get('tex_label');invlabels.append(label)
        matches=[b for b in blocks if b['label']==label]
        if len(matches)!=1:
            issues.append(f"Inventory {row.get('figure_id')}: missing or ambiguous TeX label {label}")
            continue
        b=matches[0]
        image=row.get('tex_includegraphics')
        if not isinstance(image,str) or image.strip()!=b['image']:
            issues.append(f"{label}: includegraphics source differs from inventory")
        caption=' '.join(str(row.get('caption','')).split())
        if not caption or caption!=b['caption']:
            issues.append(f"{label}: manuscript caption differs from inventory")
        if row.get('tex_file')!=b['tex_file']:
            issues.append(f"{label}: source TeX file differs from inventory")
        # Confirm the LaTeX graphic resolves to the identical inventory asset.
        # Search the figure-source directory, main-TeX directory and root.
        asset=row.get('asset')
        if not isinstance(asset,str) or not asset.strip():
            issues.append(f"{label}: missing actual source-linked inventory asset")
        else:
            expected=(root/asset).resolve()
            image=Path(b['image'])
            image_paths=[(root/b['tex_file']).parent/image, main.parent/image, root/image]
            matched=any(candidate.resolve()==expected for candidate in image_paths)
            if not expected.is_relative_to(root) or not expected.is_file() or not matched:
                issues.append(f"{label}: LaTeX image does not resolve to inventory asset")
    if len(blocks)!=len(listed):
        issues.append(f"Unreconciled figure environments: TeX {len(blocks)} vs inventory {len(listed)}")
    if len(invlabels)!=len(set(invlabels)):
        issues.append('Duplicate inventory TeX labels')
    return {'status':'BLOCKED' if issues else 'STRUCTURAL_TEX_BINDINGS_PASS_REVIEW_REQUIRED',
            'source_environments':len(blocks),'inventory_figures':len(listed),'issues':issues}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();result=audit(a.repo_root,a.inventory)
    print(json.dumps(result,indent=2));return int(bool(result['issues']))
if __name__=='__main__':raise SystemExit(main())
