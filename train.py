
import params
from UNET import *



################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay

################################### LOADING DATA TRANSVERSAAL  ##################################



################################## TRAINING  ##################################
def MSE(input,output):
    return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)


model = UNet().to(device)
optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
train_loader = DataLoader(MuizenDataset(Train_Data_001h,Train_Data_024h),batch_size=batch_size,shuffle=True,drop_last=True)
n_total_steps = len(train_loader)

model.train()

print('Starting with training...')
for epoch in range(num_epochs):  # we itereren meerdere malen over de data tot convergence?

    for i, (batch_voor, batch_na) in enumerate(tqdm(train_loader)): #wat is een handige manier om dit in te lezen?
        
        if torch.cuda.is_available(): #steek de batch in de GPU
            batch_voor=Variable(batch_voor.cuda())
            batch_na=Variable(batch_na.cuda())
        
        optimizer.zero_grad()
        predicted_batch = model(batch_voor)
        _, _, H, W = predicted_batch.shape
        batch_na = torchvision.transforms.CenterCrop([H,W])(batch_na)
        
        loss = loss_function(predicted_batch,batch_na) #vergelijk predicted na image met de echte na image
        loss.backward()
        optimizer.step()
        
        batch_voor = torchvision.transforms.CenterCrop([H,W])(batch_voor)

    print(f'Epoch: {epoch}, Loss: {loss.item()}')

##################################### SAVING THE MODEL  ##################################
model_path = "model.pth"
torch.save(model.state_dict(),model_path)
print('Model trained succesfully and saved as model.pth')
