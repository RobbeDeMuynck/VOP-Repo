from run import *

obj = LieveMuizen(batch_size=8,features=8,learning_rate=0.03)
obj.prep_data()
obj.train()
obj.test(plot=True)
