from tdqm import tdqm
def BasicTraining(Dataset=None, 
                  Model=None, 
                  Epoch=None, 
                  BatchSize=None, 
                  ValInterval=None, 
                  log=None):
    for i in tdqm(range(1000)):
        a = 1