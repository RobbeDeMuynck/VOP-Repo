from run import *

obj = LieveMuizen(batch_size=4,features=8,learning_rate=0.0001)
obj.prep_data()
obj.train()
obj.test(plot=True)
