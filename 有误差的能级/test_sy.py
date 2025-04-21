import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time


class Mynetwork(nn.Module):
    def __init__(self,input_num=1 , out_num=1,hidden_num=128):
        super().__init__()
        self.MLP=nn.Sequential(
            nn.Linear(input_num, hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num,hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num,hidden_num),
            nn.ELU(),       
            nn.Linear(hidden_num,hidden_num),
            nn.ELU(),   
            nn.Linear(hidden_num,out_num),
        )
        pass
    def forward(self,x):
        return self.MLP(x)+self.MLP(-x)
    
def potential(x):
    poten=2*x**2
    return poten

exe_num=10
total_time=time.time()

en_num=40
extend_num=10
epoch=200000
lr=0.01

h_bar=1
m=1
b_lap:float=-h_bar**2/(2*m)

# 同时对于库伦势函数, 取e=1, 4\pi\epsilon_0=1, E_n=-1/(2n^2)
dtype=torch.float32
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

La=-10
Lb =10
L=Lb-La  # domain length
N = 800   # number of interior points # 对时间成本来说几乎是平方量级
h :float= L / (N+1)
grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
grid=grid[1:-1].unsqueeze(-1)


diag = -2.0 / h**2 * torch.ones(N,device=device) * b_lap
off_diag = 1.0 / h**2 * torch.ones(N - 1,device=device) * b_lap


V_diag=potential(grid)
A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
eigenvalues= torch.linalg.eigvalsh(A)
real_en=eigenvalues[:en_num].detach()


#######————————————————————————————————————########    
eig_loss_list=[]
poten_loss_list=[]

for execution in range(exe_num):
    
    # torch.manual_seed(seed=42) 
    os.makedirs(f'./model_{execution+1}', exist_ok=True)
    
    model=Mynetwork().to(device=device,dtype=dtype)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=60,threshold=1e-4)
    loss_fn=nn.L1Loss()

    init_time=time.time()
    for i in range(epoch):
        optimizer.zero_grad()
        V_diag=model(grid)
        A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
        eigenvalues= torch.linalg.eigvalsh(A)
        output=eigenvalues[:en_num]
        
        loss=loss_fn(output,real_en)
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        if optimizer.param_groups[0]["lr"] <= 1.1e-8:break
        
    final_loss=loss.item()
    final_time=time.time()-init_time
    final_epoch=i+1
    
    eig_loss_list.append(final_loss)

    torch.save(model.state_dict(),f'./model_{execution+1}/model_para.pth')

    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)
    NN_poten=model(grid)
    real_poten=potential(grid)
    
    poten_loss=loss_fn(NN_poten,real_poten)
    poten_loss_list.append(poten_loss.item())
    
    NN_poten=NN_poten.cpu().detach().numpy()
    real_poten=real_poten.cpu().detach().numpy()
    grid=grid.cpu().detach().numpy()
    plt.plot(grid,NN_poten,label='NN')
    plt.plot(grid,real_poten,label='real')
    plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
    plt.legend()
    plt.savefig(f'./model_{execution+1}/potential.png')
    plt.clf()

    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)
    V_diag=model(grid)
    A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
    eigenvalues= torch.linalg.eigvalsh(A)
    output=eigenvalues[:en_num]

    plt.plot(output[:].detach().cpu().numpy(),label='NN')
    plt.plot(real_en[:].detach().cpu().numpy(),label='real')
    plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
    plt.legend()
    plt.savefig(f'./model_{execution+1}/eigenvalue.png')
    plt.clf()
    
    del model
    torch.cuda.empty_cache()
    print(f'Execution {execution+1} completed.')
    print(f'time: {time.time()-init_time:.2f}s , epoch: {final_epoch} , loss: {final_loss:.4f}')
    print('total time:',(time.time()-total_time)/60,' min') 
    
plt.plot(eig_loss_list,label='eig_loss')
plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
plt.legend()
plt.savefig(f'./eig_loss.png')
plt.clf()

plt.plot(poten_loss_list,label='poten_loss')
plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
plt.legend()
plt.savefig(f'./poten_loss.png')
plt.clf()

print(' 所有执行完成 ')