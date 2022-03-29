
import params
import MiceData
from UNET import *



################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay

################################### LOADING DATA TRANSVERSAL  ###################################
input = MiceData.Train_transversal_001h
target = MiceData.Train_transversal_024h

################################## TRAINING  ##################################
def MSE(input,output):
    return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)


model = UNet().to(device)
optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
                )
train_loader = DataLoader(
                MuizenDataset(input, target),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
                )
n_total_steps = len(train_loader)

model.train()

print('Starting with training...')
for epoch in range(num_epochs):  # we itereren meerdere malen over de data tot convergence?

    for i, (input_batch, target_batch) in enumerate(tqdm(train_loader)): #wat is een handige manier om dit in te lezen?
        
        if torch.cuda.is_available(): #steek de batch in de GPU
            input_batch = Variable(input_batch.cuda())
            target_batch = Variable(target_batch.cuda())
        
        optimizer.zero_grad()
        prediction_batch = model(input_batch)
        _, _, H, W = prediction_batch.shape
        target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)
        
        loss = loss_function(prediction_batch, target_batch) #vergelijk predicted na image met de echte na image
        loss.backward()
        optimizer.step()
        
        input_batch = torchvision.transforms.CenterCrop([H,W])(input_batch)

    print(f'Epoch: {epoch}, Loss: {loss.item()}')

##################################### SAVING THE MODEL  ##################################
model_path = "model_test.pth"
torch.save(model.state_dict(),model_path)
print('Model trained succesfully and saved as {model_path}')
