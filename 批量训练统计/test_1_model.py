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
        return self.MLP(x)
    
def potential(x,k,scale=100):
    poten=-10/x
    return poten

exe_num=10

h_bar=1
m=1
b_lap:float=-h_bar**2/(2*m)

# 同时对于库伦势函数, 取e=1, 4\pi\epsilon_0=1, E_n=-1/(2n^2)
dtype=torch.float32
device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

La=0
Lb =400
L=Lb-La  # domain length
N = 2000   # number of interior points # 对时间成本来说几乎是平方量级
h :float= L / (N+1)
grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
grid=grid[1:-1].unsqueeze(-1)

# 控制势函数的大小
l_max=2
scale=10

en_num=30
extend_num=10
epoch=200000
lr=0.01

diag = -2.0 / h**2 * torch.ones(N,device=device) * b_lap
off_diag = 1.0 / h**2 * torch.ones(N - 1,device=device) * b_lap

real_en_list=[]
for l in range(l_max):
    centrifugal_poten=-b_lap*(l+1)*l/grid**2
    V_diag=potential(grid,b_lap,scale)+centrifugal_poten
    A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
    eigenvalues= torch.linalg.eigvalsh(A)
    real_en=eigenvalues[:en_num].detach()
    real_en_list.append(real_en)


#######————————————————————————————————————########    

for execution in range(exe_num):
    
    # torch.manual_seed(seed=42) 
    os.makedirs(f'./model_{execution+1}', exist_ok=True)
    
    model=Mynetwork().to(device=device,dtype=dtype)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=60,threshold=1e-4)
    loss_fn=nn.L1Loss()

    loss_list=[]
    init_time=time.time()
    for i in range(epoch):
        optimizer.zero_grad()
        
        loss=0
        for l in range(l_max):
            centrifugal_poten=-b_lap*(l+1)*l/grid**2
            V_diag=model(grid)+centrifugal_poten
            A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
            eigenvalues= torch.linalg.eigvalsh(A)
            output=eigenvalues[:en_num]
            
            loss+=loss_fn(output,real_en_list[l])
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        scheduler.step(loss)
        if optimizer.param_groups[0]["lr"] <= 1.1e-8:break
        
    final_loss=loss.item()
    final_time=time.time()-init_time
    final_epoch=i+1

    torch.save(model.state_dict(),f'./model_{execution+1}/model_para.pth')

    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)
    grid=grid[20:]
    NN_poten=model(grid)
    real_poten=potential(grid,b_lap,scale)
    NN_poten=NN_poten.cpu().detach().numpy()
    real_poten=real_poten.cpu().detach().numpy()
    grid=grid.cpu().detach().numpy()
    plt.plot(grid,NN_poten,label='NN')
    plt.plot(grid,real_poten,label='real')
    plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
    plt.legend()
    plt.savefig(f'./model_{execution+1}/poten_20_2000.png')
    plt.clf()

    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)
    grid=grid[:20]
    NN_poten=model(grid)
    real_poten=potential(grid,b_lap,scale)
    NN_poten=NN_poten.cpu().detach().numpy()
    real_poten=real_poten.cpu().detach().numpy()
    grid=grid.cpu().detach().numpy()
    plt.plot(grid,NN_poten,label='NN')
    plt.plot(grid,real_poten,label='real')
    plt.title(f'loss={final_loss:.4f} , epoch={final_epoch} , time={final_time:.2f}s')
    plt.legend()
    plt.savefig(f'./model_{execution+1}/poten_0_20.png')
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

print(' 所有执行完成 ')