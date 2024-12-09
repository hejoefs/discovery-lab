import deepchem as dc

from rdkit import Chem
from rdkit.Chem import Descriptors as dp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

_, (df, _, _), _ = dc.molnet.load_delaney(featurizer='ECFP')
df = pd.DataFrame(data={'SMILES': df.ids, 'LogS': df.y.flatten()})

mols = [Chem.MolFromSmiles(mol) for mol in df.SMILES]

def genDescLogS(smiles):
    data = [[dp.MolLogP(mol), dp.MolWt(mol), dp.NumRotatableBonds(mol)] for mol in mols]
    return data

def aromaticAtoms(smiles):
    data = [sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])/dp.HeavyAtomCount(mol) for mol in mols]
    return data

df1 = pd.DataFrame(genDescLogS(df.SMILES), columns=['LogP', 'Weight', 'Rotatable Bonds'])
df2 = pd.DataFrame(aromaticAtoms(df.SMILES), columns=['Aromatic Proportion'])

x = pd.concat([df1, df2], axis=1)
y = df.iloc[:, 1]

plt.figure(figsize=(4.25, 8.5))
plt.subplot(2, 1, 1)
y.hist(grid=0, color='gray')

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred, c='gray', alpha=0.25)
p = np.poly1d(np.polyfit(y_test, y_pred, 1))
plt.plot(y_test, p(y_test), color='black', alpha=0.35)

plt.title('Molecular Solubility')
plt.ylabel('Predicted')
plt.xlabel('Experimental')