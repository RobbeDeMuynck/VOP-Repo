
import params
import MiceData
from UNET import *

model_path = "model_test.pth"
patience = 2

################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
num_epochs = 20
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay

################################### LOADING DATA TRANSVERSAL  ###################################
input = MiceData.Train_transversal_001h
target = MiceData.Train_transversal_024h
val_input = MiceData.Test_transversal_001h
val_target = MiceData.Test_transversal_024h

################################## TRAINING  ##################################
# def MSE(input,output):
#     return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)

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
# test_loader = DataLoader(
#                 MuizenDataset(?, ?),
#                 batch_size=batch_size,
#                 shuffle=True,
#                 )
val_loader = DataLoader(
                MuizenDataset(val_input, val_target),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
                )
# n_train = len(train_loader)

print('Starting with training...')
loss_stats = {
    'train': [],
    'val': []
    }

best_loss = np.inf
for epoch in range(num_epochs):  # we itereren meerdere malen over de data tot convergence?
    model.train()
    train_epoch_loss = 0
    
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

        train_epoch_loss += loss.item()
        
        input_batch = torchvision.transforms.CenterCrop([H,W])(input_batch)
    
    # Prevent overfitting
    with torch.no_grad():
        val_epoch_loss = 0
            
        model.eval()
        for val_input_batch, val_target_batch in val_loader:
            val_input_batch = val_input_batch.to(device)
            val_target_batch = val_target_batch.to(device)

            val_pred = model(val_input_batch)
            val_target_batch = torchvision.transforms.CenterCrop([H,W])(val_target_batch)

            val_loss = loss_function(val_pred, val_target_batch)
            val_epoch_loss += val_loss.item()
    
    loss_stats["train"].append(train_epoch_loss/len(train_loader))
    loss_stats["val"].append(val_epoch_loss/len(val_loader))
    
    if loss_stats["val"][-1] < best_loss:
        best_loss, epoch_no = loss_stats["val"][-1], epoch+1
        torch.save(model.state_dict(), model_path)
    
    print(f"""Epoch: {epoch+1}
Training loss: {loss_stats["train"][-1]};\t Validation loss: {loss_stats["val"][-1]}
Best loss: {best_loss}
Last validation losses: {loss_stats['val'][-patience:]}""")

    if np.all(np.array(loss_stats['val'][-patience:]) > best_loss):
        print(f"""
Training terminated after epoch no. {epoch+1}
Model saved as '{model_path}': version at epoch no. {epoch_no}""")

        break
    elif epoch == num_epochs-1:
        print(f"Model trained succesfully and saved as: '{model_path}'")

print(f"""Training_losses = {loss_stats["train"]}
Validation_losses = {loss_stats["val"]}""")

##################################### SAVING THE MODEL  ##################################
