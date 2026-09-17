"""Pass 14 closure for NLP Part I Figures 3, 18, 19."""
from pathlib import Path
import os,sys,json
import numpy as np, matplotlib.pyplot as plt
HERE=Path(__file__).resolve();REPO=next(p for p in HERE.parents if (p/'chapters').is_dir())
sys.path.insert(0,str(REPO/'chapters'/'_shared'))
from pass14_closure_utils import extract_pdf_figure_near_caption,save_provenance
ROOT=Path(os.environ.get('PASS14_ARTIFACT_ROOT','pass14_artifact'));CH='08_NLP_Part_I';OUT=ROOT/CH/'Figures_Production';OUT.mkdir(parents=True,exist_ok=True);prov={}
# Fig 3: real pretrained GloVe vectors; semantic neighborhood around airplane plus PCA projection.
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
wv=api.load('glove-wiki-gigaword-100')
seed=['airplane','pilot','pilots','flight','jet','aircraft','airport','helicopter','car','ship','train','runway','passenger','flying','aviation']
words=[w for w in seed if w in wv]
# Add nearest learned neighbors of airplane so the figure is determined by the pretrained embedding itself.
for w,_ in wv.most_similar('airplane',topn=12):
    if w not in words: words.append(w)
V=np.vstack([wv[w] for w in words]);Z=PCA(n_components=2,random_state=1729).fit_transform(V)
fig,ax=plt.subplots(figsize=(7.6,5.0));ax.scatter(Z[:,0],Z[:,1],s=24)
for i,w in enumerate(words): ax.text(Z[i,0]+.02,Z[i,1]+.02,w,fontsize=8)
ax.set_xlabel('PCA component 1');ax.set_ylabel('PCA component 2');ax.grid(False);fig.tight_layout();p=OUT/'Figure_003_Word_embedding_vectors_from_the_TensorFlow_Embedding_Projector_visuali.png';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
# cosine similarities explicitly recorded
sims={w:float(np.dot(wv['airplane'],wv[w])/(np.linalg.norm(wv['airplane'])*np.linalg.norm(wv[w]))) for w in words if w!='airplane'}
prov['3']={'embedding':'GloVe wiki-gigaword 100d via gensim-data','projection':'PCA(2)','words':words,'cosine_to_airplane':sims}
# Figs 18-19: exact source tables from the GPT-4 Technical Report; crop excludes printed table caption.
pdf='https://cdn.openai.com/papers/gpt-4.pdf'
p=OUT/'Figure_018_GPT_4_Prompt_demonstrating_GPT_4_s_visual_input_capability.png'
prov['18']=extract_pdf_figure_near_caption(pdf,"Table 3. Example prompt demonstrating GPT-4's visual input capability",p,band_points=520)
prov['18'].update({'source':'OpenAI GPT-4 Technical Report, Table 3'})
p=OUT/'Figure_019_Example_of_GPT_4_providing_correct_and_incorrect_answers_on_TurthfulQA.png'
prov['19']=extract_pdf_figure_near_caption(pdf,'Table 4: Example of GPT-4 giving correct and incorrect responses on TruthfulQA',p,band_points=300)
prov['19'].update({'source':'OpenAI GPT-4 Technical Report, Table 4','benchmark':'TruthfulQA'})
save_provenance(ROOT/CH/'pass14_provenance.json',prov)
