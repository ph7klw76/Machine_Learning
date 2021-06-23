"""
Created on Sat Nov  9 13:34:04 2019
generate 1-D coloumbic matrices
@author: user
"""

from rdkit.Chem import AllChem as Chem
from collections import Counter
import pandas as pd
import numpy as np
from openbabel import pybel

open_babel = True # if false, use rdkit
omit_repetition = False # omit repeated values in matrix

pd.set_option('display.width', 150)
pd.options.display.float_format = '{:.3f}'.format

empty=[0]
smiles=[]
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros
with open('E:/test.txt', 'r+') as f:
    for line in f.readlines():
        mysmiles = line.strip('\n')
        smiles.append(mysmiles)
          
datatosave=[]
for smile in smiles:
        data=[]
        matrix = []
        atoms=[]
        print(smile)
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        Chem.EmbedMolecule(mol, Chem.ETKDG())
        conf = mol.GetConformer()
        pymol = pybel.readstring('smi', smile)
        pymol.addh()
        pymol.make3D()
        n_atoms = mol.GetNumAtoms()
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if open_babel:
            xyz = [pymol.atoms[index].coords for index in range(n_atoms)]
        else:
            xyz = conf.GetPositions()
        m = np.zeros((n_atoms, n_atoms))
        for r in range(n_atoms):
          for c in range(n_atoms):
              if r == c:
                  m[r][c] = 0.5 * z[r] ** 2.4
              elif r < c:
                  m[r][c] = z[r] * z[c] / np.linalg.norm(np.array(xyz[r]) - np.array(xyz[c])) * 0.52917721092
                  if not omit_repetition:
                      m[c][r] = m[r][c]
              else:
                  continue
        syms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Counter(syms)
        bonds = Counter([bond.GetBondType().name for bond in mol.GetBonds()])
        df = pd.DataFrame(m, columns=syms, index=syms)
        print('\n')
        print('Molecule: ' + smile)
        print('Number of atoms: ' + str(n_atoms))
        print('Atom counts: ' + str(atoms))
        print('Bond counts: ' + str(bonds))
        print(df)
        print(syms)
        print(len(syms))
        z=zerolistmaker(26-len(syms))
        numpy_matrix = df.values.tolist()
        for n, i in enumerate(syms):
            if i == 'H':
                syms[n] = 1
            if i == 'C':
                syms[n] = 6
            if i == 'N':
                syms[n] = 7
            if i == 'O':
                syms[n] = 8
            if i == 'Si':
                syms[n] = 14
            if i == 'P':
                syms[n] = 15
            if i == 'Se':
                syms[n] = 34
            if i == 'S':
                syms[n] = 16
        for ii in range (26):
            if ii <len(syms):
                matrix=numpy_matrix[ii]                   
                for i in range (26):
                    if i < len(syms):
                        data= data + list([matrix[i]])
                    else:
                        data= data +empty
            else:
                emptylist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                data= data+emptylist
        data=data+z+syms
        print(data)
        datatosave.append(data)
        print(len(data))
print('test')
print(datatosave)
np.save('E:/test',datatosave)                  
