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
    poten=x*torch.sin(torch.pi*x*2/3)+x**2*2
    return poten

var_env=os.environ.get('EN_NUM')

if __name__ == '__main__':
    exe_num=100
    sigma=0.005

    total_time=time.time()
    en_num=int(var_env)*10+40
    file_name=f'f6_model_{sigma}_{en_num}'

    epoch=200000
    lr=0.01

    h_bar=1
    m=1
    b_lap:float=-h_bar**2/(2*m)

    # 同时对于库伦势函数, 取e=1, 4\pi\epsilon_0=1, E_n=-1/(2n^2)
    dtype=torch.float32
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    La=-40
    Lb =40
    L=Lb-La  # domain length
    N = 3200   # number of interior points # 对时间成本来说几乎是平方量级
    h :float= L / (N+1)
    grid=torch.linspace(La,Lb,N+2,dtype=dtype,device=device)
    grid=grid[1:-1].unsqueeze(-1)


    diag = -2.0 / h**2 * torch.ones(N,device=device) * b_lap
    off_diag = 1.0 / h**2 * torch.ones(N - 1,device=device) * b_lap
    
    V_diag=potential(grid)
    A = torch.diag(diag) + torch.diag(off_diag,diagonal=1) + torch.diag(off_diag, diagonal=-1)+torch.diag(V_diag.flatten())
    eigenvalues= torch.linalg.eigvalsh(A)
    real_en_0=eigenvalues[:en_num].detach()

    #######————————————————————————————————————########    
    eig_loss_list=[]
    pre_loss_list=[]

    potential_list=[]
    eig_list=[]
    rand_real_eig_list=[]

    for execution in range(exe_num):
        
        ####################
        rand_en=sigma*torch.randn(en_num,device=device,dtype=dtype)+1
        ####################
        real_en=real_en_0*rand_en
        real_en,_=torch.sort(real_en)
        
        rand_real_eig_list.append(real_en.detach().cpu().numpy())
        
        
        # torch.manual_seed(seed=42) 
        os.makedirs(f'./{file_name}', exist_ok=True)
        
        model=Mynetwork().to(device=device,dtype=dtype)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=50,threshold=1e-4)
        loss_fn=nn.L1Loss()
        
        pre_loss=loss_fn(real_en,real_en_0)
        pre_loss_list.append(pre_loss.item())

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
        
        eig_list.append(output.detach().cpu().numpy())
        potential_list.append(V_diag.flatten().detach().cpu().numpy())
        # torch.save(model.state_dict(),f'./{file_name}/model_para.pth')

        del model
        torch.cuda.empty_cache()
        print(f'Execution {execution+1} completed.')
        print(f'time: {time.time()-init_time:.2f}s , epoch: {final_epoch} , loss: {final_loss:.4f}')
        print('total time:',(time.time()-total_time)/60,' min') 
        

    real_en_0=real_en_0.cpu().numpy()
    plt.plot(pre_loss_list,label='pre_rand_loss')
    plt.title(f'pre_rand_eig_loss')
    plt.legend()
    plt.savefig(f'./{file_name}/pre_rand_eig_loss.png')
    plt.clf()

    plt.plot(eig_loss_list,label='eig_loss')
    plt.title(f'eig_loss')
    plt.legend()
    plt.savefig(f'./{file_name}/eig_loss.png')
    plt.clf()

    grid=torch.linspace(La,Lb,N+2)
    grid=grid[1:-1]
    real_poten=potential(grid)
    real_poten=real_poten.numpy()

    dpi=1200
    #画出势能图像的训练的标准差和平均曲线
    mean_poten=np.mean(potential_list,axis=0)
    std_poten=np.std(potential_list,axis=0)
    upper_bound=mean_poten+std_poten
    lower_bound=mean_poten-std_poten
    plt.plot(grid,mean_poten,label='mean',zorder=0,color='b',alpha=0.5)
    plt.fill_between(grid,upper_bound,lower_bound,alpha=0.2,color='b',label='std')
    plt.plot(grid,real_poten,label='real',zorder=10,color='r',alpha=0.5)
    plt.title(f'potential with error band')
    plt.legend()
    plt.savefig(f'./{file_name}/potential_with_err.png',dpi=dpi)
    plt.clf()
    
    plt.plot(grid,std_poten,label='std',zorder=0,color='b',alpha=0.5)
    plt.legend()
    plt.savefig(f'./{file_name}/potential_std.png',dpi=dpi)
    plt.clf()

    for i in range(len(potential_list)):
        plt.plot(grid,potential_list[i],zorder=i+1,color='g',alpha=0.5)
    plt.plot(grid,real_poten,label='real',zorder=i+10,color='r',alpha=0.5)
    plt.ylim(real_poten.min()-10,real_poten.max()+10)
    plt.title(f'potential band')
    plt.legend()
    plt.savefig(f'./{file_name}/potential_band.png',dpi=dpi)
    plt.clf()
    
    count=np.array([i for i in range(en_num)])
    for i in range(len(eig_list)):
        plt.plot(count,eig_list[i],zorder=i+1,color='g',alpha=0.5)
    plt.plot(count,real_en_0,label='real',zorder=i+10,color='r',alpha=0.5)
    plt.ylim(real_en_0.min()-1,real_en_0.max()+1)
    plt.title(f'eigenvalue band')
    plt.legend()
    plt.savefig(f'./{file_name}/eig_band.png',dpi=dpi)
    plt.clf()
    
    #画出eig的训练的标准差和平均曲线
    mean_eig=np.mean(eig_list,axis=0)
    std_eig=np.std(eig_list,axis=0)
    upper_bound_eig=mean_eig+std_eig
    lower_bound_eig=mean_eig-std_eig
    plt.plot(count,mean_eig,label='mean',zorder=0,color='b',alpha=0.5)
    plt.fill_between(count,upper_bound_eig,lower_bound_eig,alpha=0.2,color='b',label='std')
    plt.plot(count,real_en_0,label='real',zorder=10,color='r',alpha=0.5)
    plt.ylim(real_en_0.min()-1,real_en_0.max()+1)
    plt.title(f'eigenvalue with error band')
    plt.legend()
    plt.savefig(f'./{file_name}/eig_with_err.png',dpi=dpi)
    plt.clf()
    
    plt.plot(std_eig,label='std',zorder=0,color='b',alpha=0.5)
    plt.legend()
    plt.savefig(f'./{file_name}/eig_std.png',dpi=dpi)
    plt.clf()

    for i in range(len(rand_real_eig_list)):
        plt.plot(count,rand_real_eig_list[i],zorder=i+1,color='g',alpha=0.5)
    plt.plot(count,real_en_0,label='real',zorder=i+10,color='r',alpha=0.5)
    plt.ylim(real_en_0.min()-1,real_en_0.max()+1)
    plt.title(f'initial eigenvalue band')
    plt.legend()
    plt.savefig(f'./{file_name}/initial_eig_band.png',dpi=dpi)
    plt.clf()
    
    #画出initial_eig的训练的标准差和平均曲线
    mean_ini_eig=np.mean(rand_real_eig_list,axis=0)
    std_ini_eig=np.std(rand_real_eig_list,axis=0)
    upper_bound_ini_eig=mean_ini_eig+std_ini_eig
    lower_bound_ini_eig=mean_ini_eig-std_ini_eig
    plt.plot(count,mean_ini_eig,label='mean',zorder=0,color='b',alpha=0.5)
    plt.fill_between(count,upper_bound_ini_eig,lower_bound_ini_eig,alpha=0.2,color='b',label='std')
    plt.plot(count,real_en_0,label='real',zorder=10,color='r',alpha=0.5)
    plt.ylim(real_en_0.min()-1,real_en_0.max()+1)
    plt.title(f'initial eigenvalue with error band')
    plt.legend()
    plt.savefig(f'./{file_name}/initial_eig_with_err.png',dpi=dpi)
    plt.clf()

    print(' 所有执行完成 ')