
################################### DECLARING HYPERPARAMETERS  ##################################
import params
import MiceData
from UNET import *

num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay
plotting = False
input = MiceData.Test_transversal_001h
target = MiceData.Test_transversal_024h

############################# IMPORTING THE NEEDED FUNCTIONS  #############################

############################# LOADING THE MODEL  #############################
model_path = "model.pth"
model = UNet().to(device)
model.load_state_dict(torch.load(model_path))



############################# TESTING  #############################

test_loader = DataLoader(
                MuizenDataset(input, target),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
                )
model.eval()
print('Starting with testing...')


losses = []
for i, (input_batch,target_batch) in enumerate(tqdm(test_loader)):
    #input_batch = input_batch.view(batch_size,1,121,242)
    #target_batch = target_batch.view(batch_size,1,121,242)
    
    if torch.cuda.is_available():
        input_batch=Variable(input_batch.cuda())
        target_batch=Variable(target_batch.cuda())
    prediction_batch = model(input_batch)

    _, _, H, W = prediction_batch.shape
    target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)

    loss = loss_function(prediction_batch,target_batch) #vergelijk predicted na image met de echte na image
    losses.append(loss.item())

    if plotting == True:
        for j in range(batch_size):
            fig = plt.subplots(figsize=(20,40))
            img_pred = prediction_batch[j][0].cpu()
            img_input = input_batch[j][0].cpu()
            img_target = target_batch[j][0].cpu()
            plt.subplot(1,3,1)
            plt.imshow(img_input.detach().cpu().numpy(),cmap='bone')
            plt.title('Before injection')
            
            plt.subplot(1,3,2)
            plt.imshow(img_target.detach().numpy(),cmap='bone')
            plt.title('24 hours after injection')
            plt.subplot(1,3,3)
            plt.imshow(img_pred.detach().numpy(),cmap='bone')
            plt.title('Model prediction')
            plt.show()

print(f'The average test loss is: {np.mean(np.array(losses))}')
