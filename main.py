from run import *
losses = []
for bs in [2,4,6,8,10]:
    for lr in [0.01,0.03,.1]:
        for f in [8,16,32]:
            obj = LieveMuizen(batch_size=4,features=32)
            obj.prep_data()
            obj.train()
            losses.append((obj.test(plot=False),bs,lr,f))

print(losses)