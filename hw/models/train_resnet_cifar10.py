import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import zfpy
import numpy as np
from tqdm import tqdm
import pandas as pd
from time import perf_counter

from resnet import Bottleneck, ResNet, ResNet50


transform_train = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(32, padding=4),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

g_total_steps = 0
g_total_codec_sec = 0
g_total_transfer_sec = 0

EPOCHS = 200
tag = 'all-200e'
print(f'Running for {tag=} {EPOCHS=}')
tolerances = [1e-1, 1e-3, 1e-6, 1e-9]
rates = [4, 8, 16, 32]

def collect_gradients(model):
  grad_tensors = []
  
  #* Iterate over model parameters
  for param in model.parameters():
    #* If parameter has gradient, collect it
    if param.grad is not None:
        grad_tensors.append(param.grad.view(-1))
  
  #* Concatenate all collected gradient tensors
  collected_grads = torch.cat(grad_tensors)
  
  return collected_grads

def assign_gradients(model, collected_grads):
  #* Counter for iterating over collected gradients
  idx = 0
  
  #* Iterate over model parameters
  for param in model.parameters():
    #* If parameter has gradient, assign the corresponding values from collected gradients
    if param.grad is not None:
      #* Calculate the number of elements in the parameter tensor
      num_elements = param.grad.numel()
      #* Assign values from collected gradients to the parameter's gradient tensor
      param.grad.data = collected_grads[idx:idx+num_elements].view(param.grad.size())
      #* Move to the next batch of collected gradients
      idx += num_elements


def train_one_step(zfp_mode, variable, inputs, labels, model, criterion, optimizer, device):
  global g_total_steps
  global g_total_codec_sec
  global g_total_transfer_sec
  
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, labels)
  loss.backward()
  
  if zfp_mode == 'lossless':
    gradients = collect_gradients(model)
    #* Measure transfer time of uncompressed gradients
    transfer_start = perf_counter()
    gradients = gradients.cpu().numpy()
    gradients = torch.from_numpy(gradients).to(device)
    transfer_end = perf_counter()
    g_total_transfer_sec += transfer_end - transfer_start
    assign_gradients(model, gradients)
    
    optimizer.step()
    return loss.item()
  
  gradients = collect_gradients(model)
  #* Transfer gradients to host for compression
  transfer_start = perf_counter()
  gradients = gradients.cpu().numpy()
  transfer_end = perf_counter()
  g_total_transfer_sec += transfer_end - transfer_start
  
  #* CODEC
  codec_start = perf_counter()
  if zfp_mode == 'precision':
    gradients = zfpy.compress_numpy(gradients, precision=variable)
  elif zfp_mode == 'accuracy':
    gradients = zfpy.compress_numpy(gradients, tolerance=variable)
  elif zfp_mode == 'rate':
    gradients = zfpy.compress_numpy(gradients, rate=variable)
  gradients = zfpy.decompress_numpy(gradients)
  codec_end = perf_counter()
  g_total_codec_sec += codec_end - codec_start
  
  #* Transfer gradients back to device
  transfer_start = perf_counter()
  gradients = torch.from_numpy(gradients).to(device)
  transfer_end = perf_counter()
  g_total_transfer_sec += transfer_end - transfer_start
  
  assign_gradients(model, gradients)

  optimizer.step()
  return loss.item()

def validate(model, testloader, criterion):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return correct/total * 100

if __name__ == '__main__':
  # zfp_modes = ['lossless', 'precision', 'accuracy', 'rate']
  #* Get GPU id from command line
  gpu_id = int(sys.argv[1])
  #* Set GPU
  torch.cuda.set_device(gpu_id)
  device = f'cuda:{gpu_id}'
  #* Get mode from command line
  zfp_mode = sys.argv[2]
  
  variables = tolerances if zfp_mode == 'accuracy' else rates
  if zfp_mode == 'lossless':
    variables = [0]
  
  records = []
  for variable in variables:
    model = ResNet50(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    
    start = perf_counter()
    print(f'Running for {zfp_mode=}: {variable=}')
    for epoch in tqdm(range(EPOCHS)):
      losses = []
      running_loss = 0
      for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
      
        loss = train_one_step(
          zfp_mode, variable, inputs, labels, model, criterion, optimizer, device)
        running_loss += loss
        losses.append(loss)
        
        g_total_steps += 1
        
        # if i%100 == 0 and i > 0:
        #   print(f'epoch={epoch+1} batch={i} | loss: {running_loss / 100:.3f}')
        #   running_loss = 0.0

      avg_loss = sum(losses)/len(losses)
      scheduler.step(avg_loss)
      
    end = perf_counter()
    duration_sec = end - start
      
    print('Training Done')
    accuracy = validate(model, testloader, criterion)
    records.append({
      'mode': zfp_mode, 
      'variable': variable, 
      'accuracy': accuracy, 
      'total_duration_sec': duration_sec,
      'total_codec_sec': g_total_codec_sec,
      'total_transfer_sec': g_total_transfer_sec,
      'total_steps': g_total_steps,
      'epochs': EPOCHS,
    })
    
    print(f'Accuracy: {accuracy:.3f}%')
    print(f'Duration: {duration_sec:.3f} sec')
    print(f'Codec time: {g_total_codec_sec:.3f} sec')
    print(f'Transfer time: {g_total_transfer_sec:.3f} sec')
    print('-'*50)
    
    duration_sec = 0
    g_total_codec_sec = 0
    g_total_transfer_sec = 0
    g_total_steps = 0
    
    f = f'./results/{tag}/resnet_{zfp_mode}.csv'
    pd.DataFrame(records).to_csv(f, index=False)
    print(f'Results saved to {f}')
    f = f'./models/resenet_{zfp_mode}-{variable}_{tag}.pt'
    torch.save(model.state_dict(), f)
    print(f'Model saved to {f}')
  
  print('All done')
  
  
