url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv"

from rdkit import Chem
from rdkit.Chem import Descriptors as dp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv(url)
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

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred, c='gray', alpha=0.25)
p = np.poly1d(np.polyfit(y_test, y_pred, 1))
plt.plot(y_test, p(y_test), color='black', alpha=0.35)

plt.title('Aqueous Solubility')
plt.ylabel('Predicted')
plt.xlabel('Experimental')