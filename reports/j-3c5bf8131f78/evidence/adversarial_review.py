from types import SimpleNamespace
from unittest.mock import patch
import torch
from sglang.srt.models.deepseek_common import v32_mixin
from sglang.srt.models.deepseek_common.attention_backend_handler import AttnForwardMethod
from sglang.srt.models.deepseek_common.hardware_backend import deepseek_v2_npu_mixin

class Model(v32_mixin.DeepseekV32ModelMixin): pass
m=Model(); m.use_dsa=True; m.config=SimpleNamespace(num_hidden_layers=8); m.start_layer=3; m.end_layer=7
for uses, skip_start, skip_end, expect_in, expect_out in [
 (False,True,True,False,False),(True,False,False,False,False),
 (True,True,False,True,False),(True,False,True,False,True),
 (True,True,True,True,True)]:
 def skip(_, layer, ss=skip_start, se=skip_end):
  return ss if layer==3 else se if layer==7 else False
 with patch.object(v32_mixin,'dsa_layer_skips_topk',side_effect=skip):
  assert m.dsa_stage_requires_input_topk(uses)==expect_in
  assert m.dsa_stage_must_forward_topk(uses)==expect_out
x=torch.randn(2,3,device='cuda')
with patch.object(v32_mixin,'get_dsa_index_topk',return_value=19): y=m.empty_dsa_topk(x)
assert y.shape==(0,19) and y.dtype==torch.int32 and y.device==x.device

class A(deepseek_v2_npu_mixin.DeepseekV2NPUAttentionMixin): pass
a=A()
for method, needs_topk in [(AttnForwardMethod.MHA_NPU,False),(AttnForwardMethod.MLA_NPU,False),(AttnForwardMethod.DSA_NPU,True)]:
 seen={}
 def prep(*args): seen['prep']=args; return ('state',)
 def core(*args): seen['core']=args; return 'ok'
 with patch.object(deepseek_v2_npu_mixin,'_npu_attention_functions',return_value={method:(prep,core)}):
  state=a.forward_npu_prepare(method,'p','h','b','z','s','topk')
  assert len(seen['prep'])==(7 if needs_topk else 6)
  assert (seen['prep'][-1]=='topk') is needs_topk
  assert a.forward_npu_core(method,state)=='ok' and seen['core']==(a,'state')
print('PASS adversarial truth table, NPU routing signatures, GPU empty top-k')
print('gpu',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).gcnArchName)
