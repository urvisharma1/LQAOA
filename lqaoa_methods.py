from pennylane import qaoa
from pennylane import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from math import pi
import random
import torch
import copy
import seaborn as sns
import time

# @title n node graph
n_qubits=15
p=0.55 #probability of edge formation
wires = n_qubits

#Graph 
#Uncomment only if u want to see what the graph looks like
#graph = nx.erdos_renyi_graph(n_qubits, p,seed=52)#, seed=9,shots=1000
# Print out the generated graph G
#plt.figure(figsize=(8, 6))
#nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500, font_size=12, edge_color='gray')
#plt.title(f"{n_qubits} Graph", fontsize=16)
#plt.show()
#cost_h, mixer_h = qml.qaoa.maxcut(graph)

time_taken = {}

# @title Circuit
dev = qml.device('default.qubit', wires=wires)#shots not added, Why? Refer limitations section.

def qaoa_layer(gamma, beta):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(beta, mixer_h)

def circuit(gammas, betas):
    for w in range(wires):
        qml.Hadamard(wires=w)
    for gamma, beta in zip(gammas, betas):
        qaoa_layer(gamma, beta)

@qml.qnode(dev, interface='torch')
def cost(gammas, betas):
    circuit(gammas, betas)
    return qml.expval(cost_h)

@qml.qnode(dev, interface='torch')
def final_probs(gammas, betas):
    circuit(gammas, betas)
    return qml.probs(wires=range(wires))

#params
gammas = [torch.tensor(random.uniform(0, 2*pi), requires_grad=True)]
betas = [torch.tensor(random.uniform(0, pi), requires_grad=True)]
#defined to ensure that gammas and betas remain unchanged
gamma0 = copy.deepcopy(gammas)
beta0 = copy.deepcopy(betas)
lr=0.02
decay_factor=0.1
n_steps = 250
new_layer =100 
depth=round(n_steps/new_layer)

def optimizer(gammas,betas,lr):
  return torch.optim.Adam(gammas + betas, lr)

'''
#Random Model Seeds, to check if optimizer performs differently.
#This wasn't particularly useful
def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)  # Python random
  torch.manual_seed(seed)  # PyTorch random
'''
def QAOA_optimize(gammas, betas, n_steps,new_layer,lr,seed=None):
    '''
    QAOA optimization
    '''
    steps = []
    cost_values = []
    start_time = time.time()

    # Optimization loop
    for step in range(n_steps):
        optimizer(gammas,betas,lr).zero_grad()  
        loss = cost(gammas, betas)  
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(gammas + betas, max_norm=1.0)
        optimizer(gammas,betas,lr).step() 
        steps.append(step)
        cost_values.append(loss.item())
        if step % new_layer == 0:
            print(f"Step {step}: Cost = {loss.item():.4f}")
            # Uncomment: if u need to print mid optimizationn to check if the params are getting updated
            #print(f"Current gammas: {gammas}")
            #print(f"Current betas: {betas}")
    end_time = time.time()
    time_taken= end_time - start_time
    print(f"Optimization completed in {time_taken:.4f} seconds")
    print(f"Optimal gammas: {gammas}")
    print(f"Optimal betas: {betas}")

    return steps, cost_values, time_taken, gammas, betas

#  parameters for layer 1
#'depth' Layers are getting trained together
depth=round(n_steps/new_layer)
g_n=[copy.deepcopy(gamma0)[0] for _ in range(depth)]
t_n=[copy.deepcopy(beta0)[0] for _ in range(depth)]


#continued'''
def LQAOA1(gammas, betas, n_steps, new_layer,lr,seed=None):
    '''
    check new_gamma and new_beta to see how the params are updated
    '''
    steps = []
    cost_values = []

    start_time = time.time()

    for step in range(n_steps):
        optimizer(gammas,betas,lr).zero_grad()
        loss = cost(gammas, betas)
        loss.backward()
        #normalize
        torch.nn.utils.clip_grad_norm_(gammas + betas, max_norm=1)
        optimizer(gammas,betas,lr).step()

        steps.append(step)
        cost_values.append(loss.item())
        if (step + 1) % new_layer == 0:
            #freeezelayers
            for gamma, beta in zip(gammas, betas):
                gamma.requires_grad = False
                beta.requires_grad = False
            # EVERY new_layer steps, add a new layer 
            new_gamma = torch.tensor(gammas[-1].item(), requires_grad=True)
            new_beta = torch.tensor(betas[-1].item(), requires_grad=True)
            gammas.append(new_gamma)
            betas.append(new_beta)

        if step % new_layer == 0:
            print(f"Step {step}: Cost = {loss.item():.4f}")
            # Uncomment: if u need to print mid optimizationn to check freezing and adding
            #print(f"Current gammas: {gammas}")
            #print(f"Current betas: {betas}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Optimization completed in {time_taken:.4f} seconds")
    print(f"Optimal gammas: {gammas}")
    print(f"Optimal betas: {betas}")
    return steps, cost_values, time_taken,gammas,betas


#randomised'''
def randomC(gammas, betas, n_steps, new_layer,lr,seed=None):
    '''
    check new_gamma and new_beta to see how the params are updated
    '''
    steps = []
    cost_values = []
    start_time = time.time()

    for step in range(n_steps):
        optimizer(gammas,betas,lr).zero_grad()
        loss = cost(gammas, betas)
        loss.backward()
        #normalize
        torch.nn.utils.clip_grad_norm_(gammas + betas, max_norm=1)
        optimizer(gammas,betas,lr).step()

        steps.append(step)
        cost_values.append(loss.item())
        if (step + 1) % new_layer == 0:
            #freeezelayers
            for gamma, beta in zip(gammas, betas):
                gamma.requires_grad = False
                beta.requires_grad = False
            # EVERY new_layer steps, add a new layer
            new_gamma = torch.tensor(random.uniform(0, 2*pi), requires_grad=True)
            new_beta = torch.tensor(random.uniform(0, pi), requires_grad=True)
            gammas.append(new_gamma)
            betas.append(new_beta)

        if step % new_layer == 0:
            print(f"Step {step}: Cost = {loss.item():.4f}")
            # Uncomment: if u need to print mid optimizationn to check freezing and adding
            #print(f"Current gammas: {gammas}")
            #print(f"Current betas: {betas}")

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Optimization completed in {time_taken:.4f} seconds")
    print(f"Optimal gammas: {gammas}")
    print(f"Optimal betas: {betas}")
    return steps, cost_values, time_taken,gammas,betas

#BACKWARD
def LQAOA2(gammas, betas, n_steps,new_layer,lr,seed=None):
    '''
    check new_gamma and new_beta to see how the params are updated
    '''
    steps = []
    cost_values = []
    start_time = time.time()
    for step in range(n_steps):
        optimizer(gammas, betas,lr).zero_grad() 
        loss = cost(gammas, betas)  
        loss.backward()  
        #torch.nn.utils.clip_grad_norm_(gammas + betas, max_norm=1)
        optimizer(gammas, betas,lr).step()  
        steps.append(step)
        cost_values.append(loss.item())
        if (step + 1) % new_layer == 0:
            # Freeze previous layers except for the last 'window_size' layers,g0.t6,3
            for gamma, beta in zip(gammas, betas):
                gamma.requires_grad = False
                beta.requires_grad = False
            #add a new layer w updated params
            new_gamma = torch.tensor(gammas[-1].item() / 2, requires_grad=True)
            new_beta = torch.tensor(betas[-1].item() / 2, requires_grad=True)
            gammas.append(new_gamma)
            betas.append(new_beta)


        if step % new_layer == 0:
            print(f"Step {step}: Cost = {loss.item():.4f},lr={lr}")
            # Uncomment: if u need to print mid optimizationn to check freezing and adding
            #print(f"Current gammas: {gammas}")
            #print(f"Current betas: {betas}")

    end_time = time.time()
    time_taken= end_time - start_time
    print(f"Optimization completed in {time_taken:.4f} seconds")
    print(f"Optimal gammas: {gammas}")
    print(f"Optimal betas: {betas}")

    return steps,cost_values,time_taken,gammas,betas

#train

def LQAOA_adapt(gammas, betas, n_steps, new_layer, lr, decay_factor, convergence_threshold=1e-3,seed=None):
    '''
    check thee gammas list
    '''
    steps = []
    cost_values = []
    prev_cost = float("inf")  
    depth = 1
    max_depth = round(n_steps / new_layer)
    start_time = time.time()
    optimizer = torch.optim.Adam(gammas + betas, lr)
    for step in range(n_steps):
        optimizer.zero_grad()  
        loss = cost(gammas, betas)  
        loss.backward()  
        # Clip gradients to prevent instability #normalization
        torch.nn.utils.clip_grad_norm_(gammas + betas, max_norm=1.0)
        optimizer.step()  
        steps.append(step)
        cost_values.append(loss.item())
        relative_change = abs((loss.item() - prev_cost) / prev_cost)

        if step % 10 == 0 and step > 0:
            if relative_change < convergence_threshold:
                lr *= decay_factor
                print(f"Reducing learning rate to: {lr:.8f}")

        # add new layers iff convergence within layer is reached
        if (step + 1) % new_layer == 0 and depth < max_depth:
            if relative_change < convergence_threshold:  # Only add layers if not improving much
                depth += 1
                for gamma, beta in zip(gammas[:-1], betas[:-1]):
                    gamma.requires_grad = False
                    beta.requires_grad = False
                # a slight random variation
                new_gamma = torch.tensor(gammas[-1].item() + random.uniform(-0.1, 0.1), requires_grad=True)
                new_beta = torch.tensor(betas[-1].item() + random.uniform(-0.1, 0.1), requires_grad=True)
                gammas.append(new_gamma)
                betas.append(new_beta)
                optimizer = torch.optim.Adam(gammas + betas, lr)

        if step % new_layer == 0:
            print(f"Step {step}: Cost = {loss.item():.4f}, lr={lr}")

    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.4f} seconds")
    print(f"Final Depth: {depth}")
    return steps, cost_values, end_time - start_time, gammas, betas

convergence_threshold = 1e-3 
#plot_probability_distribution(g_a, t_a, 'Adaptive Optimization')


#Find the max probability bitstring to find cuts
all_bitstrings = [format(i, f'0{wires}b') for i in range(2**wires)]

def MaxProb_bitstring(gammas, betas, all_bitstrings):
    probabilities = final_probs(gammas, betas).detach().numpy()
    most_frequent_bitstring = all_bitstrings[np.argmax(probabilities)]
    return most_frequent_bitstring

#Function to count Cuts
def count_cuts(graph, bitstring):
    '''
    This function counts the edges that separate two sets of nodes based on a bitstring,
    where each node's group (Group 1 or Group 2) is determined by whether its corresponding bit is '0' or '1'.
    The result is the number of edges that "cut" the graph into two disjoint parts.
    '''
    G1 = [i for i in range(len(bitstring)) if bitstring[i] == '0']
    G2 = [i for i in range(len(bitstring)) if bitstring[i] == '1']
    cuts = 0
    for u, v in graph.edges():
        if (u in G1 and v in G2) or (u in G2 and v in G1):
            cuts += 1
    return cuts

#Plot Functions
def plot_boxplots(cost_data, cuts_data, labels, cost_filename, cuts_filename):
    sns.set(style="whitegrid")
    cost_lr1olumns = [[sublist[i] for sublist in cost_data] for i in range(5)]
    cuts_columns = [[sublist[i] for sublist in cuts_data] for i in range(5)]

    #Plot Cost
    plt.figure(figsize=(12, 8))
    plt.boxplot(cost_lr1olumns, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='darkblue'),
                whiskerprops=dict(color='darkblue', linewidth=1.5),
                capprops=dict(color='darkblue', linewidth=2),
                flierprops=dict(marker='o', color='red', markersize=5),
                medianprops=dict(color='purple', linewidth=3, linestyle='-', marker='|', markersize=10))
    plt.xlabel('Optimization Method', fontsize=14)
    plt.ylabel('Cost Values', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(cost_filename, dpi=300)
    plt.show()
    plt.close()

    # Plot Cuts
    plt.figure(figsize=(12, 8))
    plt.boxplot(cuts_columns, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                whiskerprops=dict(color='darkgreen', linewidth=1.5),
                capprops=dict(color='darkgreen', linewidth=2),
                flierprops=dict(marker='o', color='red', markersize=5),
                medianprops=dict(color='purple', linewidth=3, linestyle='-', marker='|', markersize=10))
    plt.xlabel('Optimization Method', fontsize=14)
    plt.ylabel('Cuts Values', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(cuts_filename, dpi=300)
    plt.show()
    plt.close()


#MAIN
cost_list_seeds=[]
cuts_list_seeds=[]
bitstring_methods=[]
seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
for seed in seeds:
    graph = nx.erdos_renyi_graph(n_qubits, p, seed=seed)
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    '''
    #Uncomment if you want to see each graph corresponding to a seed in seeds
    plt.figure(figsize=(8, 6))
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500, font_size=12, edge_color='gray')
    plt.title(f"Graph generated with seed {seed}", fontsize=16)
    plt.show()
    plt.close()
    '''
    steps_n, cost_n, time_taken["Normal"],g_n,t_n= QAOA_optimize([copy.deepcopy(gamma0)[0] for _ in range(depth)],[copy.deepcopy(beta0)[0] for _ in range(depth)], n_steps, new_layer,lr,seed)
    steps_lr1,cost_lr1,time_taken["LQAOA1"],g_c,t_c = LQAOA1(copy.deepcopy(gamma0),copy.deepcopy(beta0), n_steps, new_layer,lr,seed)
    steps_r,cost_r,time_taken["Random"],g_r,t_r = randomC(copy.deepcopy(gamma0),copy.deepcopy(beta0), n_steps, new_layer,lr,seed)
    steps_lr2,cost_lr2,time_taken["LQAOA2"],g_b,t_b = LQAOA2(copy.deepcopy(gamma0),copy.deepcopy(beta0), n_steps,new_layer,lr,seed)
    steps_a, cost_a, time_taken['Adaptive'], g_a, t_a = LQAOA_adapt(copy.deepcopy(gamma0),copy.deepcopy(beta0),n_steps,new_layer,lr,decay_factor,convergence_threshold,seed)
    cost_list_seeds.append([cost_r[-1],cost_n[-1],cost_a[-1],cost_lr2[-1],cost_lr1[-1]])#RNABC
    cuts_list_seeds.append([count_cuts(graph, MaxProb_bitstring(g_r, t_r, all_bitstrings)),count_cuts(graph, MaxProb_bitstring(g_n, t_n, all_bitstrings)),count_cuts(graph, MaxProb_bitstring(g_a, t_a, all_bitstrings)),count_cuts(graph, MaxProb_bitstring(g_b, t_b, all_bitstrings)),count_cuts(graph, MaxProb_bitstring(g_c, t_c, all_bitstrings))])
    bitstring_methods.append([MaxProb_bitstring(g_r, t_r, all_bitstrings),MaxProb_bitstring(g_n, t_n, all_bitstrings),MaxProb_bitstring(g_a, t_a, all_bitstrings),MaxProb_bitstring(g_b, t_b, all_bitstrings),MaxProb_bitstring(g_c, t_c, all_bitstrings)])


#Plot
labels = ['Random', 'QAOA', 'LQAOA m=n=1 + Adapt', 'LQAOA m=n=2', 'LQAOA m=n=1']
plot_boxplots(cost_list_seeds, cuts_list_seeds, labels, 'cost_boxplot.png', cuts_filename='cuts_boxplot.png')

# Plotting cost converegence graph
plt.plot(steps_a, cost_a, label='m=n=1 Adaptive', color='green')
plt.plot(steps_n, cost_n, label='Traditional QAOA', color='k')
plt.plot(steps_lr1, cost_lr1, label='m=n=1', color='red')
plt.plot(steps_lr2, cost_lr2, label='m=n=2', color='blue')
plt.plot(steps_r, cost_r, label='m=n=2', color='blue')
steps = np.linspace(0, n_steps, num=new_layer+1)
x_labels= [f"Step {int(step)} " for step in (steps)]
plt.xticks(steps, x_labels, rotation=90)
plt.xlabel("Steps")
plt.ylabel("Cost")
plt.title(f"Cost Function vs. Optimization Steps")
subtitle_text = f'n_qubits: {n_qubits}, n_steps: {n_steps}, new_layer: {new_layer}'
plt.text(0, 1, subtitle_text, ha='left', va='top', fontsize=8, transform=plt.gca().transAxes)
plt.legend()
#plt.savefig('CostBox.png', dpi=300)
plt.show()
plt.close()

# @title  time comparison
avg_times = {method: np.mean(times) for method, times in time_taken.items()}
methods = list(avg_times.keys())
avg_times_values = list(avg_times.values())
plt.figure(figsize=(8, 5))
plt.bar(methods, avg_times_values, color=['orange','k', 'red','blue', 'green'])
#check the color order
#print("order:",bitstring_methods)
plt.title('Average Time Comparison', fontsize=16)
subtitle_text = f'n_qubits: {n_qubits}, n_steps: {n_steps}, new_layer: {new_layer}'
plt.text(0, 1, subtitle_text, ha='left', va='top', fontsize=8, transform=plt.gca().transAxes)
plt.xlabel('Methods')
plt.ylabel('Average Time Taken (seconds)')
#plt.savefig("avg_timeBox.png")
plt.show()
plt.close()
