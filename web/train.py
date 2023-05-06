from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, metrics
from xgboost import XGBClassifier
import scikitplot as skplt
import matplotlib.pyplot as plt
import shap
import xgboost
shap.initjs()
import streamlit as st
import fragmenter


# Create the header of the app
st.title("🌺 FragmentInsight Trainer 🌺")
st.write("Welcome to FragmentInsight, a web app that allows you to train and explore LTP Bioactivity Models.")

# upload the data
st.subheader("Upload your data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

st.sidebar.title("Options")

st.sidebar.subheader("Choose a classification method")
classification = st.sidebar.selectbox("Classification", ["Random Forest", "XGBoost"])

st.sidebar.subheader("Choose the fragmentation depth")
max_depth = st.sidebar.slider("max fragmentation depth", 0, 10, 7, 1)

st.sidebar.subheader("Choose threshold for activity")
threshold = st.sidebar.selectbox("Activity threshold (in uM)", ["100 uM", "10 uM", "1 uM", "0.1 uM"])

st.sidebar.subheader("Cross Validation Folds")
cv = st.sidebar.slider("Cross Validation Folds", 1, 5, 3, 1)

st.sidebar.subheader("Choose the top k fragments")
top_k = st.sidebar.slider("top_k = ", 0, 2048, 500, 1)

# remove made with streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.MolWt(mol)

def run_fragrank_frag(smis, threhsold_mw=250):
    fragments = []
    bar = st.progress(0)
    for n, smi in enumerate(smis):
        bar.progress(n/len(smis), text=f"Fragmenting Molecules:: {n}/{len(smis)}")
        fragments.append(fragmenter.frag_rec(smi, max_depth=max_depth))
    fragments = [list(filter(lambda x: get_mw(x) < threhsold_mw, fragment)) for fragment in fragments]
    bar.empty()
    unique_subs = set()
    for subs in chain.from_iterable(fragments):
        unique_subs.add(subs)
    G = nx.Graph()
    for sub in unique_subs:
        G.add_node(sub)
    for i in range(len(smis)):
        for j in range(len(fragments[i])):
            for k in range(j+1, len(fragments[i])):
                sub1 = fragments[i][j]
                sub2 = fragments[i][k]
                if G.has_edge(sub1, sub2):
                    G[sub1][sub2]['weight'] += 1
                else:
                    G.add_edge(sub1, sub2, weight=1)
    pr = nx.pagerank(G)
    return pr

def featurize(smis, top_fragments_mols):
    X_vec = []
    for smi in tqdm(smis, desc="Featurizing"):
        mol = Chem.MolFromSmiles(smi)
        X_vec.append(np.array([mol.HasSubstructMatch(i) for i in top_fragments_mols]).astype(int))
    return np.array(X_vec) 

def model(clf, X_train, X_test, y_train, y_test, top_fragments_mols):
    X_train = featurize(X_train, top_fragments_mols)
    X_test = featurize(X_test, top_fragments_mols)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    st.write("Recall: ", metrics.recall_score(y_test, y_pred))
    st.write("Precision: ", metrics.precision_score(y_test, y_pred))
    st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    y_probas = clf.predict_proba(X_test)
    plt = skplt.metrics.plot_roc(y_test, y_probas)
    st.pyplot(plt.figure)
    plt = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    st.pyplot(plt.figure)
    plt = skplt.metrics.plot_precision_recall(y_test, y_probas)
    st.pyplot(plt.figure)

# read csv file if uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["canonical_smiles"] = df["canonical_smiles"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    df["mw"] = df["canonical_smiles"].apply(lambda x: get_mw(x))
    df = df[df["mw"] < 800]
    st.write(df.sample(5, random_state=0))
    
    # create a train test split button
    if st.button("Start Training"):
        sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=0)
        df = df[df[threshold].notna()]
        X = df["canonical_smiles"].values
        y = df[threshold].values
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            st.write(f"Fold {i+1}/{sss.get_n_splits(X, y)}: Top K: {top_k}")
            if classification == "Random Forest":
                clf = RandomForestClassifier()
            elif classification == "XGBoost":
                clf = XGBClassifier()
            pr = run_fragrank_frag(X[train_index][y[train_index] == 1.0])
            top_fragments = sorted(pr, key=pr.get, reverse=True)[:top_k]
            top_fragments_mols = [Chem.MolFromSmarts(i) for i in top_fragments]
            model(clf, X[train_index], X[test_index], y[train_index], y[test_index], top_fragments_mols)
