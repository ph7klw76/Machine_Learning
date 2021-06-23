"""
Created on Wed Apr 22 21:22:39 2020
Machine learning assisyed Quantum Molecular Dyanmics simulation
@author: KL WOON
"""



from ase import Atoms
from ase.optimize import BFGS
import torchani
from ase import units
from ase.md.langevin import Langevin
def atomicmass(a):
    if a=='H':
        a=1
    if a=='C':
        a=6
    if a=='N':
        a=7
    if a=='O':
        a=8
    if a=='S':
        a=16
    return a        
mymolecule=[]
myposition=[]
myatom=[] 
fp=open(r'C:\Users\Nurul\Desktop\FYP\Data\TADF in Zeonex\ACRXTN-2.xyz', 'r') 
# Path Of The File
for i,line in enumerate(fp):
    line1=line.strip('\n')
    line1=line.split()   
if i>1:
        mymolecule.append(atomicmass(line1[0]))
        myatom.append(line1[0])
        atomposition=(float(line1[1]),float(line1[2]),float(line1[3]))
        myposition.append(atomposition)
    if i==0:
        line1=line.strip('\n')
        line1=int(line1)
        noatom=int(line1)
molecule = Atoms(numbers=mymolecule,positions=myposition)
print(myatom)        
calculator = torchani.models.ANI2x().ase()
molecule.set_calculator(calculator)
print("Begin minimizing...")
opt = BFGS(molecule)
opt.run(fmax=0.0001)
print()
Z=molecule.get_positions()
path=r'C:\Users\Nurul\Desktop\FYP\Data\TADF in Zeonex\ACRXTN-2.txt' 
#Path Of The File 
with open(path, "r") as ff: # or "rb"
    file_data = ff.read()
# And then:
raw = open(path, "w")
def storexyz(noatom,Z,m):
    myposition2=[]
    myposition2.append(str(noatom))
    with open(path, "a") as f:
    f.write(str(noatom)+"\n")
    with open(path, "a") as f:
        f.write(m)   
    myposition2.append(" ")
    for i,line in enumerate(Z):
        a,b,c=line[0],line[1],line[2]
        a=str(a)
        b=str(b)
        c=str(c)
        atomposition=str(myatom[i])+"   "+a+"   "+b+"   "+c
        with open(path, "a") as f:
            f.write(atomposition+"\n") 
        myposition2.append(atomposition)
    return myposition2
def printenergy(a=molecule):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    Z=molecule.get_positions()
    m=('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK) '
                   'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)+"\n") 
    storexyz(noatom,Z,m)
    return
dyn = Langevin(molecule, 1 * units.fs, 300 * units.kB, 0.2)  #The temperature is changeable 
dyn.attach(printenergy, interval=50)
print("Beginning dynamics...")
mydata=printenergy()
dyn.run(10000) 
