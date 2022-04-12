from run import *

################################## k-fold crossvalidation (k=6)  ##########################################

for k in range(6):
    obj = LieveMuizen(test_mouse=k)
    obj.prep_data()
    obj.train()
    obj.test()

for k in range(6):
    obj = LieveMuizen(test_mouse=k,view='sagittal')
    obj.prep_data()
    obj.train()
    obj.test()

for k in range(6):
    obj = LieveMuizen(test_mouse=k,view='coronal')
    obj.prep_data()
    obj.train()
    obj.test()