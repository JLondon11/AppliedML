import copy
import json
import tempfile
import unittest
from pathlib import Path
from verify_tex_figure_bindings import audit

class FigureBindings(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'chapter').mkdir()
        (self.root/'chapter'/'main.tex').write_text('\\input{figs}\n')
        (self.root/'chapter'/'images').mkdir()
        (self.root/'chapter'/'images'/'observed.png').write_bytes(b'test fixture image, not real data')
        (self.root/'chapter'/'figs.tex').write_text(r'''\begin{figure}[ht]
\includegraphics[width=.8\linewidth]{images/observed.png}
\caption{Original measured results of {model A}.}
\label{fig:original}
\end{figure}''')
        self.inv={'manuscript_tex':'chapter/main.tex','figures':[{'figure_id':'6.1',
                'tex_label':'fig:original','tex_includegraphics':'images/observed.png',
                'asset':'chapter/images/observed.png',
                'caption':'Original measured results of {model A}.','tex_file':'chapter/figs.tex'}]}
        self.path=self.root/'inventory.json'
    def check(self,inv=None):
        self.path.write_text(json.dumps(inv if inv is not None else self.inv))
        return audit(self.root,self.path)
    def test_genuine_bound_source_passes_structural_only(self):
        self.assertEqual(self.check()['status'],'STRUCTURAL_TEX_BINDINGS_PASS_REVIEW_REQUIRED')
    def test_missing_figure_block_fails(self):
        (self.root/'chapter'/'figs.tex').write_text('')
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_mismatched_caption_fails(self):
        data=copy.deepcopy(self.inv);data['figures'][0]['caption']='Invented results'
        self.assertEqual(self.check(data)['status'],'BLOCKED')
    def test_mismatched_asset_fails(self):
        data=copy.deepcopy(self.inv);data['figures'][0]['tex_includegraphics']='other.png'
        self.assertEqual(self.check(data)['status'],'BLOCKED')
    def test_figure_not_in_inventory_fails(self):
        data=copy.deepcopy(self.inv);data['figures']=[]
        self.assertEqual(self.check(data)['status'],'BLOCKED')
    def test_duplicate_inventory_label_fails(self):
        data=copy.deepcopy(self.inv);data['figures'].append(copy.deepcopy(data['figures'][0]))
        self.assertEqual(self.check(data)['status'],'BLOCKED')
    def test_missing_inventory_fails_closed(self):
        self.assertEqual(audit(self.root,self.path)['status'],'BLOCKED')
    def test_unrelated_inventory_asset_fails(self):
        data=copy.deepcopy(self.inv);data['figures'][0]['asset']='chapter/main.tex'
        self.assertEqual(self.check(data)['status'],'BLOCKED')
    def test_missing_actual_figure_asset_fails(self):
        (self.root/'chapter'/'images'/'observed.png').unlink()
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_missing_latex_fails(self):
        (self.root/'chapter'/'main.tex').unlink()
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_unsafe_include_fails(self):
        (self.root/'chapter'/'main.tex').write_text(r'\input{../../secret}')
        self.assertEqual(self.check()['status'],'BLOCKED')
    def test_comments_not_counted(self):
        p=self.root/'chapter'/'figs.tex'
        p.write_text('%'+p.read_text().replace('\n','\n%')+'\n')
        self.assertEqual(self.check()['status'],'BLOCKED')
if __name__=='__main__':unittest.main()
