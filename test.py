
################################### DECLARING HYPERPARAMETERS  ##################################
import params
from UNET import *
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay
plotting = False
############################# IMPORTING THE NEEDED FUNCTIONS  #############################

############################# LOADING THE MODEL  #############################
model_path = "model.pth"
model = UNet().to(device)
model.load_state_dict(torch.load(model_path))



############################# TESTING  #############################

test_loader = DataLoader(MuizenDataset(Test_Data_001h,Test_Data_024h),batch_size=batch_size,shuffle=True,drop_last=True)
model.eval()
print('Starten met testen...')


losses = []
for i, (batch_voor,batch_na) in enumerate(tqdm(test_loader)):
    #batch_voor = batch_voor.view(batch_size,1,121,242)
    #batch_na = batch_na.view(batch_size,1,121,242)
    
    if torch.cuda.is_available():
        batch_voor=Variable(batch_voor.cuda())
        batch_na=Variable(batch_na.cuda())
    predicted_batch = model(batch_voor)

    _, _, H, W = predicted_batch.shape
    batch_na = torchvision.transforms.CenterCrop([H,W])(batch_na)

    loss = loss_function(predicted_batch,batch_na) #vergelijk predicted na image met de echte na image
    losses.append(loss.item())
    if plotting==True:
        for j in range(batch_size):
            fig = plt.subplots(figsize=(20,40))
            afb_pred = predicted_batch[j][0].cpu()
            afb_voor = batch_voor[j][0].cpu()
            afb_na = batch_na[j][0].cpu()
            plt.subplot(1,3,1)
            plt.imshow(afb_voor.detach().cpu().numpy(),cmap='bone')
            plt.title('Voor injectie')
            
            plt.subplot(1,3,2)
            plt.imshow(afb_na.detach().numpy(),cmap='bone')
            plt.title('Na injectie')
            plt.subplot(1,3,3)
            plt.imshow(afb_pred.detach().numpy(),cmap='bone')
            plt.title('Predictie')
            plt.show()

print(f'The average test loss is: {np.mean(np.array(losses))}')
