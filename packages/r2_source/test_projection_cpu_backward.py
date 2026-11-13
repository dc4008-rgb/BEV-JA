"""CPU mathematical checks of a scoped fixed-road-grid backward candidate."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('projection_candidate_test',HERE/'projection_cpu_backward.py')
projection=importlib.util.module_from_spec(spec);spec.loader.exec_module(projection)



class ProjectionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)

 def test_double_gradcheck_bilinear_zero_padding_fixed_nonrectangular_grid(self):
  g=torch.Generator().manual_seed(1923)
  x=torch.rand((1,3,3,4),generator=g,dtype=torch.float64,requires_grad=True)
  grid=torch.tensor([[[[-1.3,-.8],[-.5,.1],[.8,-.4]],[[.1,1.2],[1.,1.],[-.75,.73]]]],dtype=torch.float64)
  self.assertTrue(torch.autograd.gradcheck(lambda value:projection.cpu_backward_rgb_sample(value,grid),(x,),eps=1e-6,atol=1e-5,rtol=1e-4))

 def test_fixed_vjp_matches_cpu_native_exact_and_repeats(self):
  gen=torch.Generator().manual_seed(812)
  x=torch.rand((1,3,7,11),generator=gen,requires_grad=True)
  grid=torch.rand((1,13,17,2),generator=gen)*3.-1.5
  output=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
  grad=torch.randn(output.shape,generator=gen)
  expected,=torch.autograd.grad(output,x,grad_outputs=grad)
  values=[]
  for _ in range(5):
   replica=x.detach().clone().requires_grad_(True)
   y=projection.cpu_backward_rgb_sample(replica,grid)
   self.assertTrue(torch.equal(y,output))
   value,=torch.autograd.grad(y,replica,grad_outputs=grad)
   self.assertTrue(torch.equal(value,expected));values.append(value)
  self.assertTrue(all(torch.equal(values[0],v) for v in values[1:]))

 def test_learnable_grid_rejected(self):
  x=torch.ones(1,3,4,4,requires_grad=True)
  grid=torch.zeros(1,2,2,2,requires_grad=True)
  with self.assertRaisesRegex(ValueError,'learnable grid'):
   projection.cpu_backward_rgb_sample(x,grid)



if __name__=='__main__':unittest.main()
