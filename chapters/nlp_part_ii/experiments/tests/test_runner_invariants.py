#!/usr/bin/env python3
"""Offline invariants only. Passing tests is not experimental certification."""
import importlib.util, tempfile, unittest, json
from pathlib import Path
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
def module(name):
    path=BASE/name/"run.py"
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

imdb=module("imdb_bert")
doc=module("docvqa_layoutlmv3")

class IMDb(unittest.TestCase):
    def test_split(self):
        labels=[0]*100+[1]*100
        tr,va=imdb.split_stratified(labels,.1,42)
        self.assertEqual((len(tr),len(va)),(180,20))
        self.assertFalse(set(tr)&set(va))
        self.assertEqual(sum(labels[i] for i in va),10)
        self.assertEqual((tr,va),imdb.split_stratified(labels,.1,42))

class DocVQA(unittest.TestCase):
    def test_anls(self):
        self.assertEqual(doc.anls("Invoice","invoice"),1)
        self.assertEqual(doc.anls("xyz","abc"),0)
    def test_answer_spans(self):
        self.assertEqual(doc.answer_word_span(["Total","$361.80"],["$361.80"]),(1,1))
        self.assertIsNone(doc.answer_word_span(["Total"],["$361.80"]))
    def test_split_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError,"Document leakage"):
            doc.assert_disjoint({"train":[{"id":"1","document_id":"X"}],
                                 "validation":[{"id":"2","document_id":"X"}],
                                 "test":[{"id":"3","document_id":"Y"}]})
    def test_bbox_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new("RGB",(16,16),"white").save(root/"page.png")
            item={"id":"1","document_id":"x","image":"page.png","question":"Total?",
                  "words":["total"],"boxes":[[0,0,1000,1000]],"answers":["total"]}
            p=root/"test.jsonl";p.write_text(json.dumps(item)+"\n")
            self.assertEqual(len(doc.load_manifest(p)),1)
            item["boxes"]=[[0,0,1001,1000]]
            p.write_text(json.dumps(item)+"\n")
            with self.assertRaisesRegex(ValueError,"invalid OCR box"):
                doc.load_manifest(p)

if __name__=="__main__":unittest.main()
