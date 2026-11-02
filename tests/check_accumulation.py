"""Two-rank accumulation must match one reference full-batch optimizer update."""
import torch
from accelerate import Accelerator

a=Accelerator(gradient_accumulation_steps=2)
model=torch.nn.Linear(2,1,bias=False)
with torch.no_grad():model.weight.fill_(.1)
reference=torch.nn.Linear(2,1,bias=False).to(a.device)
reference.load_state_dict(model.state_dict())
optimizer=torch.optim.Adam(model.parameters(),lr=.01)
ref_optimizer=torch.optim.Adam(reference.parameters(),lr=.01)
model,optimizer=a.prepare(model,optimizer)
x=torch.arange(24,dtype=torch.float32,device=a.device).reshape(12,2)/24
y=x.sum(-1,keepdim=True)*.7
optimizer.zero_grad()
for micro in range(2):
    start=(micro*a.num_processes+a.process_index)*3
    with a.accumulate(model):
        loss=(model(x[start:start+3])-y[start:start+3]).square().mean()
        a.backward(loss)
        optimizer.step();optimizer.zero_grad()
    if micro==0:
        assert not optimizer.state,'Optimizer stepped before accumulation finished'
ref_loss=(reference(x)-y).square().mean();ref_loss.backward();ref_optimizer.step()
torch.testing.assert_close(a.unwrap_model(model).weight,reference.weight,atol=1e-7,rtol=1e-6)
assert all(float(s['step'])==1 for s in optimizer.state.values())
a.wait_for_everyone()
if a.is_main_process: print('TWO_GPU_ACCUMULATION_MATCHES_GLOBAL_BATCH_12')
a.end_training()
