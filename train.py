
import params
from UNET import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay

################################### LOADING DATA TRANSVERSAAL  ##################################

Train_voor = []
Train_na = []

path = pathlib.Path('processed').parent
for timestamp in ["-001h", "024h"]:
     for mouse in ["M03", "M04", "M05", "M06", "M07"]:
         if timestamp == "-001h":
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_voor.append(nib.load(path_ct).get_fdata())
         else: 
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_na.append(nib.load(path_ct).get_fdata())           


Train_Data_001h = []
Train_Data_024h = []

for mouse in Train_voor:
    for i in range(mouse.shape[-1]):
        Train_Data_001h.append(mouse[:,:,i])


for mouse in Train_na:
    for i in range(mouse.shape[-1]):
         Train_Data_024h.append(mouse[:,:,i])


Test_Data_001h = []
Test_Data_024h = []

for timestamp in ["-001h", "024h"]:
     mouse = "M08"
     path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
     ct = nib.load(path_ct).get_fdata()
     for i in range(ct.shape[-1]):
        if timestamp == "-001h":
            Test_Data_001h.append(ct[:,:,i])
        else:
            Test_Data_024h.append(ct[:,:,i])

#Train_Data_001h (1210 slices)
#Train_Data_024h (1210 slices)
#Test_Data_001h  (242 slices)
#Test_Data_024h  (242 slices)





################################## TRAINING  ##################################
def MSE(input,output):
    return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)


model = UNet().to(device)
optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
loss_function = nn.MSELoss()
train_loader = DataLoader(MuizenDataset(Train_Data_001h,Train_Data_024h),batch_size=batch_size,shuffle=True,drop_last=True)
n_total_steps = len(train_loader)

model.train()

print('Starten met trainen...')
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
