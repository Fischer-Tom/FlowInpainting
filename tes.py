import json

import numpy as np
import torch
from imagelib.display_tools import load_net
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from utils.loss_functions import EPE_Loss
import json as js
import seaborn as sns

sns.set_style('dark')
def millions(x, pos):
    'The two args are the value and tick position'
    if x < 1_000_000:
        return '%1.0fK' % (x * 1e-3)
    return '%1.0fM' % (x * 1e-6)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
def plot_runs(weights,names, EPE):

    fig, ax = plt.subplots()

    colour = ['m', 'b', 'r', 'g', 'k', 'y', 'c']
    scats = []
    for weight,name,epe,col in zip(weights,names,EPE,colour):

        scats.append(ax.scatter(weight, epe,marker='X',s=80,c=col))

    plt.legend(scats,names,scatterpoints=1,fontsize=20)

    #plt.bar(names, weights)#, palette=['grey' if (x<max(weights)) else 'red' for x in weights])

    #plt.legend()
    """
    for i, name in enumerate(names):
        ax.annotate(name, (weights[i]+3_000, EPE[i]),c='k')#,c=colour[i])
    """

    #ax.get_xaxis().set_major_formatter(
    #    matplotlib.ticker.FuncFormatter(millions))
    #plt.xlim(-90_000,9_000_000)
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    plt.xlabel("Weights",fontsize=20)
    plt.xscale('log')
    plt.ylabel("Sintel EPE",fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_vs(names, EPE,EPE_2):

    #fig, ax = plt.subplots()
    """
    colour = ['m', 'b', 'r', 'g', 'k', 'y', 'c']
    scats = []
    for weight,name,epe,col in zip(weights,names,EPE,colour):

        scats.append(ax.scatter(weight, epe,marker='X',s=80,c=col))

    plt.legend(scats,names,scatterpoints=1)
    """
    width = 0.35
    #EPE = np.delete(EPE, [0, 2])
    #EPE_2 = np.delete(EPE_2, [0, 2])
    #names = np.delete(names, [0, 2])
    X_axis = np.arange(len(names))
    plt.bar(X_axis, EPE,width=width,label='Sintel')#, palette=['grey' if (x<max(weights)) else 'red' for x in weights])
    plt.bar(X_axis+width,EPE_2,width=width, label='FlyingThings3D')
    plt.xticks(X_axis+width / 2,names,fontsize=15)
    plt.legend(loc='best')
    """
    for i, name in enumerate(names):
        ax.annotate(name, (weights[i]+3_000, EPE[i]),c='k')#,c=colour[i])
    """

    #ax.get_xaxis().set_major_formatter(
    #    matplotlib.ticker.FuncFormatter(millions))
    #plt.xlim(-90_000,9_000_000)
    #plt.xlabel("Weights")
    plt.ylabel("Test Loss",fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_runtime(weights, timings):

    fig, ax1 = plt.subplots()
    colour = ['m', 'b', 'r', 'g', 'k', 'y', 'c']
    scats = []
    names = list(weights.keys())
    weights = np.array(list(weights.values()))
    timings = list(timings.values())
    columns = ["names", "weights", "timings"]
    data1 = weights#[10, 15, 7, 12, 9]
    data2 = timings#[5, 8, 10, 6, 12]
    categories = names

    # Create the bar plot
    bar_width = 0.4
    offset = np.arange(len(names))

    # Create the bar plot
    ax1.bar(offset-0.5*bar_width, weights, width=bar_width, align='edge', color='lightblue',label='Weights')

    #ax1 = sns.barplot(x=categories, y=data1, color='skyblue', label='Bar Plot 1', width=0.4)

    # Create a twin axes
    ax2 = ax1.twinx()

    # Plot the second bar plot on the twin axes
    #sns.barplot(x=categories, y=data2, color='orange', ax=ax2, alpha=0.7, label='Bar Plot 2', width=0.4)
    ax2.bar(offset+0.5*bar_width, timings, width=bar_width,color='orange',align='edge', label='Timings')
    # Set y-axis labels for each bar plot
    ax1.set_ylabel('Weights',fontsize=20)
    ax2.set_ylabel('Runtime in ms', fontsize=20)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylim(10**(5),10**(9.5))
    ax2.set_ylim(10**0,10**(5.5))
    #ax1.legend(handles, labels)

    # Set title
    plt.xticks(offset + bar_width / 2, names)
    for t in ax1.get_xticklabels():  # get_xticklabels will get you the label objects, same for y
        t.set_fontsize(16)
    plt.xlim(-0.5*bar_width,4+1.5*bar_width)
    plt.tight_layout()

    plt.savefig('RunTimeVsWeight.png')

def plot_mask_gener(weights, timings):

    fig, ax1 = plt.subplots()
    names = list(weights.keys())
    loss_5 = [0.5,0.65,5.5,6, 2.0]
    loss_1 = [0.8,0.9,7.0,8.5, 3.0]
    # Create the bar plot
    bar_width = 0.4
    offset = np.arange(len(names))

    # Create the bar plot
    ax1.bar(offset - 0.5 * bar_width, loss_1, width=bar_width, align='edge', color='lightblue', label='1%')

    # ax1 = sns.barplot(x=categories, y=data1, color='skyblue', label='Bar Plot 1', width=0.4)

    # Create a twin axes
    # Plot the second bar plot on the twin axes
    # sns.barplot(x=categories, y=data2, color='orange', ax=ax2, alpha=0.7, label='Bar Plot 2', width=0.4)
    ax1.bar(offset + 0.5 * bar_width, loss_5, width=bar_width, color='orange', align='edge', label='5%')
    # Set y-axis labels for each bar plot
    ax1.set_ylabel('Sintel EPE', fontsize=20)

    # ax1.legend(handles, labels)

    # Set title
    plt.xticks(offset + bar_width / 2, names)
    for t in ax1.get_xticklabels():  # get_xticklabels will get you the label objects, same for y
        t.set_fontsize(16)
    plt.xlim(-0.5 * bar_width, 4 + 1.5 * bar_width)
    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig('MaskGener.png')



nets = ["DB_InpaintingNet_5"]#, "DTM+","FlowNetS", "WGAIN", "PD"]
names = nets
prefix = '/home/fischer/FlowInpainting/Experiments/'

ds = 'Geo'
EPE = []


from dataset.FlyingThings import FlyingThingsDataset
from dataset.Sintel import SintelDataset

from torchvision import transforms

dataset = FlyingThingsDataset
sintel_dataset = SintelDataset
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 4}

# Datasets and Loaders
val_dataset = dataset("", 0.95,presmooth=False, mode='test', type='IP')
sintel_val_dataset = sintel_dataset("/home/fischer/FlowInpainting/dataset/Sintel", 0.95, presmooth=False, mode='test', type="IP")
test_loader = torch.utils.data.DataLoader(val_dataset, **params)
validation_loader = torch.utils.data.DataLoader(sintel_val_dataset, **params)

"""
run_data = pd.DataFrame(columns=[])

for idx, name in enumerate(json_files):
    with open(os.path.join(dir,name)) as json_file:
        json = js.load(json_file)
        run_name = name.split('-')[1]
        ids = [id_ for [_,id_,_] in json]
        loss = [loss_ for [_,_,loss_] in json]
        run_data['index'] = ids
        run_data[run_name] = loss[-1]
    EPE.append(loss[-1])

"""
for net_name in nets:
    net = load_net(net_name)
    #weights.append(sum(p.numel() for p in net.parameters()))

    running_loss = 0.0
    iterations = 0
    running_timings = 0.0
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if ds == 'Geo':
            for i, sample in enumerate(test_loader):
                sample = [samp.cuda() for samp in sample]

                I1, I2 = sample[0:2]
                Mask = sample[2]
                Flow = sample[3]# / 100.0
                Masked_Flow = sample[-1]# / 100.0
                # Query Model
                #r = (1 - Mask) * torch.randn_like(Masked_Flow)
                #Condition = torch.cat((I1, Masked_Flow, Mask), dim=1)

                start.record()
                #predict_flow = net.sample(batch_size=1, stop_at_unet_number=2, cond_images=Condition[0:1, ::],
                #       inpaint_images=Flow[0:1, ::], inpaint_masks=Mask[0:1, 0, ::].bool(), cond_scale=5.)
                #predict_flow = net.G(I1, Mask, Masked_Flow, r)
                predict_flow = net(I1, Mask, Masked_Flow)
                batch_risk = EPE_Loss(predict_flow, Flow)#net.get_loss(predict_flow, Flow)
                end.record()
                torch.cuda.synchronize()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
                running_timings += start.elapsed_time(end)
                if i > 1:
                    break

        else:
            for i, sample in enumerate(test_loader):
                sample = [samp.cuda() for samp in sample]

                I1, I2 = sample[0:2]
                Flow = sample[-1]
                # Query Model
                start.record()

                predict_flow = net(I1, I2)
                batch_risk = net.get_loss(predict_flow, Flow)
                end.record()
                torch.cuda.synchronize()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
                if i > 150:
                    break
        print(running_loss / iterations)
        #timings.append(running_timings / iterations)
        EPE.append(running_loss / iterations)

#np.savetxt('EPE.csv',np.asarray(EPE),delimiter=',')
#np.savetxt('timings3.csv',np.asarray(timings),delimiter=',')
#np.savetxt('weights3.csv',np.asarray(weights),delimiter=',')

#EPE = np.loadtxt('EPE.csv',delimiter=',')
#EPE_2 = np.loadtxt('EPE_FT.csv',delimiter=',')

#with open('timings.json', 'r') as f:
#    timings = json.load(f)

#with open('weights.json','r') as f:
#    weights = json.load(f)

#print(timings)
#print(weights)
#plot_mask_gener(weights, timings)
