
################################### DECLARING HYPERPARAMETERS  ##################################
import params
import MiceData
from UNET import *

num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay
patience = params.patience
features = params.features

plotting = True
# input = MiceData.Test_coronal_001h
# target = MiceData.Test_coronal_024h
input = MiceData.Test_transversal_001h
target = MiceData.Test_transversal_024h
# input = MiceData.Train_transversal_001h
# target = MiceData.Train_transversal_024h


############################# IMPORTING THE NEEDED FUNCTIONS  #############################

############################# LOADING THE MODEL  #############################
model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
model = UNet(features).to(device)
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
    prediction_batch = model(input_batch)[0]
    after_conv1_batch = model(input_batch)[1]
    after_pool1_batch = model(input_batch)[2]
    after_conv2_batch = model(input_batch)[3]
    after_pool2_batch = model(input_batch)[4]
    after_conv3_batch = model(input_batch)[5]
    after_pool3_batch = model(input_batch)[6]
    after_conv4_batch = model(input_batch)[7]
    after_pool4_batch = model(input_batch)[8]
    after_bottle_batch = model(input_batch)[9]
    after_decoder1_batch = model(input_batch)[10]
    after_decoder2_batch = model(input_batch)[11]
    after_decoder3_batch = model(input_batch)[12]
    after_decoder4_batch = model(input_batch)[13]


    _, _, H, W = prediction_batch.shape
    target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)

    loss = loss_function(prediction_batch,target_batch) #vergelijk predicted na image met de echte na image
    losses.append(loss.item())

    if plotting == True and i%80 == 0:
        for j in range(batch_size):
            img_pred = prediction_batch[j][0].cpu()
            img_input = input_batch[j][0].cpu()
            img_target = target_batch[j][0].cpu()
            img_conv1 = after_conv1_batch[j][0].cpu()
            img_conv2 = after_conv2_batch[j][0].cpu()
            img_conv3 = after_conv3_batch[j][0].cpu()
            img_conv4 = after_conv4_batch[j][0].cpu()
            img_pool1 = after_pool1_batch[j][0].cpu()
            img_pool2 = after_pool2_batch[j][0].cpu()
            img_pool3 = after_pool3_batch[j][0].cpu()
            img_pool4 = after_pool4_batch[j][0].cpu()
            img_bottle = after_bottle_batch[j][0].cpu()
            img_decoder1 = after_decoder1_batch[j][0].cpu()
            img_decoder2 = after_decoder2_batch[j][0].cpu()
            img_decoder3 = after_decoder3_batch[j][0].cpu()
            img_decoder4 = after_decoder4_batch[j][0].cpu()
            fig = plt.subplots(figsize=(20,40))
            plt.subplot(2,4,1)
            plt.imshow(img_input.detach().cpu().numpy(),cmap='viridis')
            plt.title('Before injection')
            plt.subplot(2,4,2)
            plt.imshow(img_target.detach().numpy(),cmap='viridis')
            plt.title('24 hours after injection with contrast enhancement')
            plt.subplot(2,4,3)
            plt.imshow(img_pred.detach().numpy(),cmap='viridis')
            plt.title('Model prediction')
            plt.subplot(2,4,4)
            plt.imshow(img_conv1.detach().numpy(),cmap='viridis')
            plt.title('after conv1')
            plt.subplot(2,4,5)
            plt.imshow(img_pool1.detach().numpy(),cmap='viridis')
            plt.title('after pool1')
            plt.subplot(2,4,6)
            plt.imshow(img_conv2.detach().numpy(),cmap='viridis')
            plt.title('after conv2')
            plt.subplot(2,4,7)
            plt.imshow(img_pool2.detach().numpy(),cmap='viridis')
            plt.title('after pool2')
            plt.subplot(2,4,8)
            plt.imshow(img_conv3.detach().numpy(),cmap='viridis')
            plt.title('after conv3')
            plt.show()


            fig = plt.subplots(figsize=(20,40))
            plt.subplot(2,4,1)
            plt.imshow(img_pool3.detach().cpu().numpy(),cmap='viridis')
            plt.title('after pool3')
            plt.subplot(2,4,2)
            plt.imshow(img_conv4.detach().numpy(),cmap='viridis')
            plt.title('after conv4')
            plt.subplot(2,4,3)
            plt.imshow(img_bottle.detach().numpy(),cmap='viridis')
            plt.title('after bottleneck')
            plt.subplot(2,4,4)
            plt.imshow(img_decoder1.detach().numpy(),cmap='viridis')
            plt.title('after decoder1')
            plt.subplot(2,4,5)
            plt.imshow(img_decoder2.detach().numpy(),cmap='viridis')
            plt.title('after decoder2')
            plt.subplot(2,4,6)
            plt.imshow(img_decoder3.detach().numpy(),cmap='viridis')
            plt.title('after decoder3')
            plt.subplot(2,4,7)
            plt.imshow(img_decoder4.detach().numpy(),cmap='viridis')
            plt.title('after decoder4')
            plt.subplot(2,4,8)
            plt.imshow(img_target.detach().numpy(),cmap='viridis')
            plt.title('24 hours after injection with contrast enhancement')
            plt.show()
    # if plotting == True and i%80 == 0:
    #      for j in range(batch_size):
    #          fig = plt.subplots(figsize=(20,40))
    #          img_pred = prediction_batch[j][0].cpu()
    #          img_input = input_batch[j][0].cpu()
    #          img_target = target_batch[j][0].cpu()
    #          plt.subplot(3,1,1)
    #          plt.imshow(img_input.detach().cpu().numpy(),cmap='viridis')
    #          plt.title('Before injection')
    #          plt.subplot(3,1,2)
    #          plt.imshow(img_target.detach().numpy(),cmap='viridis')
    #          plt.title('24 hours after injection with contrast enhancement')
    #          plt.subplot(3,1,3)
    #          plt.imshow(img_pred.detach().numpy(),cmap='viridis')
    #          plt.title('Model prediction')
    #          plt.show()
print(f'The average test loss is: {np.mean(np.array(losses))}')
