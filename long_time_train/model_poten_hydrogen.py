import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time
# torch.manual_seed(seed=42) 
os.makedirs('./model_para', exist_ok=True)

class Mynetwork(nn.Module):
    def __init__(self,input_num=1 , out_num=1,hidden_num=64):
        super().__init__()
        self.MLP=nn.Sequential(
            nn.Linear(input_num, hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num,hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num,hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num,out_num),
        )
        pass
    def forward(self,x):
        return self.MLP(x)/x
    
def potential(x,k,l,scale=100):
    poten=-1/x*scale
    return poten-k*(l+1)*l/x**2

def energy(en_level,l,k,scale,device,dtype):
    en_list=[scale**2/(4*k*(i+1)**2) for i in range(en_level)]
    en_list=torch.tensor(en_list,device=device,dtype=dtype)
    return en_list

if __name__ == '__main__':
    #需要适当减小,x_M=sqrt(h_bar/(m*w)*(2n+1)),考虑要>10x_M,700sqrt(h_bar/\omega*m)
    h_bar=1
    m=1
    b_lap:float=-h_bar**2/(2*m)

    # 同时对于库伦势函数, 取e=1, 4\pi\epsilon_0=1, E_n=-1/(2n^2)
    dtype=torch.float64
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    La=0
    Lb =5
    L=Lb-La  # domain length
    N = 4000   # number of interior points # 对时间成本来说几乎是平方量级
    h :float= L / (N+1)
    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)
    # 角量子数
    l=0
    # 控制势函数的大小
    scale=100
    en_num=5
    epoch=1000
    lr=0.01
    real_en=energy(en_num,l,b_lap,scale,device,dtype)

    model=Mynetwork().to(device=device,dtype=dtype)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=20,threshold=1e-4)
    loss_fn=nn.L1Loss()
    # Construct the tridiagonal matrix A
    diag = -2.0 / h**2 * torch.ones(N,device=device) * b_lap
    off_diag = 1.0 / h**2 * torch.ones(N - 1,device=device) * b_lap

    V_diag=potential(grid,b_lap,l,scale)
    A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
    eigenvalues= torch.linalg.eigvalsh(A)
    print('initial eigenvalues check:')
    print(eigenvalues[:en_num])
    
    loss_list=[]
    init_time=time.time()
    for i in range(epoch):
        optimizer.zero_grad()
        V_diag=model(grid)
        A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
        eigenvalues= torch.linalg.eigvalsh(A)
        '''output=NN(input)
        diag=torch.diag(output.flatten())'''
        
        print('\nepoch:',i)
        print(eigenvalues[:en_num])
        print(real_en)
        
        output=eigenvalues[:en_num]
        loss=loss_fn(output,real_en)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        if i%10==0:
            print(f'epoch:{i},loss:{loss},time:{time.time()-init_time},lr:{optimizer.param_groups[0]["lr"]}')
        if (i+1)%100==0:torch.save(model.state_dict(),f'./model_para/model_para_{lr}.pth')
        scheduler.step(loss)
        if optimizer.param_groups[0]["lr"] <= 1.1e-8:break
    print('terminal epoch: ',i+1)
    torch.save(model.state_dict(),f'./model_para/model_para_{lr}.pth')

        