"""V2 evaluator adapter for the legacy KNO Navier--Stokes evaluator."""
from pathlib import Path
import sys
import ast
import importlib.util
import inspect
import json
import os
import traceback
from typing import Any, Dict, Mapping

_BASE_EVALUATOR_DIR = None
for _parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    _base_dir = _parent / 'benchmarks' / 'evaluators' / 'base'
    if _base_dir.is_dir():
        _BASE_EVALUATOR_DIR = _base_dir
        break
if _BASE_EVALUATOR_DIR is None:
    raise FileNotFoundError('Could not locate repository evaluators/base directory.')
sys.path.insert(0, str(_BASE_EVALUATOR_DIR))
from base_evaluator import *


def _repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / 'benchmarks' / 'datasets').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate repository root containing benchmarks/datasets.')


candidate_env = os.environ.get("candidate_env")
stardard_name = os.environ.get("stardard_name")

def _candidate_model_path() -> Path:
    if not candidate_env:
        raise RuntimeError("candidate_env is not configured")
    candidate_path = os.environ.get(candidate_env)
    if not candidate_path:
        raise RuntimeError(f"{candidate_env} is not configured")
    return Path(candidate_path).expanduser().resolve()

def _reference_model_path() -> Path:
    return (
        _repo_root()
        / "benchmarks"
        / "standard"
        / stardard_name
    ).resolve()


CANDIDATE_DIR = _candidate_model_path()
REFERENCE_DIR = _reference_model_path()

def check_files(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['required:model.py','required:dataset.py','required:utils.py','required:train.py','required:config.yaml','syntax:model.py','syntax:dataset.py','syntax:utils.py','syntax:train.py','syntax:config.yaml']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        required = ('model.py','dataset.py','utils.py','train.py','config.yaml')
        for name in required:
            path = (CANDIDATE_DIR / name).resolve()
            update_check(current_index, path.is_file(), str(path)); current_index += 1
        for name in required[:-1]:
            path = (CANDIDATE_DIR / name).resolve()
            try:
                text = path.read_text(encoding='utf-8'); ast.parse(text, filename=str(path)); ok = bool(text.strip()); detail = f'lines={len(text.splitlines())}'
            except Exception as exc:
                ok = False; detail = f'{type(exc).__name__}: {exc}'
            update_check(current_index, ok, detail); current_index += 1
        try:
            import yaml
            path = (CANDIDATE_DIR / 'config.yaml').resolve()
            value = yaml.safe_load(path.read_text(encoding='utf-8'))
            root = value.get('root') if isinstance(value, dict) else None
            value = root if isinstance(root, dict) else value
            ok = isinstance(value, dict); detail = f'top_level={sorted(value)}' if ok else 'config.yaml must contain a mapping'
        except Exception as exc:
            ok = False; detail = f'{type(exc).__name__}: {exc}'
        update_check(current_index, ok, detail); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_config_content(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['data:mat_source','data:mat_variable','data:fixed_split','data:time_window','data:spatial_shape','data:flattened_tensors','data:dataloaders','data:float32']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import yaml
        def find_value(value, names):
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key in set(names): return item
                for item in value.values():
                    found = find_value(item, names)
                    if found is not None: return found
            return None
        config_path = (CANDIDATE_DIR / config.get('config_name','config.yaml')).resolve()
        value = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        root = value.get('root') if isinstance(value, dict) else None
        loaded = root if isinstance(root, dict) else value
        names = config.get('source_files', ('dataset.py','train.py','config.yaml'))
        source = '\n'.join((CANDIDATE_DIR / name).resolve().read_text(encoding='utf-8') for name in names if (CANDIDATE_DIR / name).resolve().is_file())
        lower, compact = source.lower(), ''.join(source.lower().split())
        data_file = config.get('data_file','NavierStokes_V1e-5_N1200_T20.mat')
        path = str(find_value(loaded, ('path','file_name')) or '')
        train_samples = find_value(loaded, ('train_samples','ntrain')); val_samples = find_value(loaded, ('val_samples','ntest','nval'))
        input_steps = find_value(loaded, ('input_steps','t_in')); output_steps = find_value(loaded, ('output_steps','t_out')); spatial_shape = find_value(loaded, ('spatial_shape',))
        downsample_x = find_value(loaded, ('downsamplex',)); downsample_y = find_value(loaded, ('downsampley',))
        train_slice = any(t in compact for t in ('raw[:self.train_samples]','raw[:ntrain','data[:1000','[:1000'))
        val_slice = any(t in compact for t in ('raw[-self.val_samples:]','raw[-ntest:','data[-200:','[-200:'))
        update_check(current_index, data_file in source and data_file in path, path); current_index += 1
        update_check(current_index, 'loadmat' in lower and ('mat_key' in lower or "['u']" in compact or '["u"]' in compact), 'MAT variable u'); current_index += 1
        update_check(current_index, train_samples == 1000 and val_samples == 200 and train_slice and val_slice, f'train={train_samples} val={val_samples}; first 1000 / last 200'); current_index += 1
        update_check(current_index, input_steps == 10 and output_steps == 10, f'input={input_steps} output={output_steps}'); current_index += 1
        runtime_spatial_shape = (
            spatial_shape is None
            and downsample_x == 1
            and downsample_y == 1
            and 'self.spatial_shape' in lower
            and 'raw.shape[1]' in compact
            and 'raw.shape[2]' in compact
            and 'height' in lower
            and 'width' in lower
        )
        shape_ok = spatial_shape in ([64,64],(64,64)) or '64*64' in compact or runtime_spatial_shape
        shape_detail = f'spatial_shape={spatial_shape}' if not runtime_spatial_shape else 'spatial_shape=(height,width) derived from raw shape with unit downsampling'
        update_check(current_index, shape_ok, shape_detail); current_index += 1
        update_check(current_index, 'reshape' in lower and 'pos' in lower and any(t in lower for t in ('fx','history','"x"')), 'fx/history [N,4096,10], target [N,4096,10], pos [4096,2]'); current_index += 1
        update_check(current_index, 'dataloader' in lower and 'shuffle=true' in compact and 'shuffle=false' in compact, 'shuffle train only; batch_size=8'); current_index += 1
        update_check(current_index, 'float32' in lower, 'vorticity tensors use float32'); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_train_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['train:defaults','train:adamw','train:step_lr','train:teacher_forcing_10_step_mse','train:autoregressive_validation','train:validation_interval','train:best_relative_l2','train:checkpoint_name','train:checkpoint_schema','train:entrypoint']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import yaml
        def find_value(value, names):
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key in set(names): return item
                for item in value.values():
                    found = find_value(item, names)
                    if found is not None: return found
            return None
        source = '\n'.join((CANDIDATE_DIR / name).resolve().read_text(encoding='utf-8') for name in ('train.py','utils.py','config.yaml') if (CANDIDATE_DIR / name).resolve().is_file())
        value = yaml.safe_load((CANDIDATE_DIR / config.get('config_name','config.yaml')).resolve().read_text(encoding='utf-8')); root = value.get('root') if isinstance(value,dict) else None; loaded = root if isinstance(root,dict) else value
        lower, compact = source.lower(), ''.join(source.lower().split())
        expected = ((find_value(loaded,('epochs',)),500),(find_value(loaded,('batch_size',)),8),(find_value(loaded,('learning_rate','lr')),1e-3),(find_value(loaded,('weight_decay',)),1e-5),(find_value(loaded,('scheduler_step_size','step_size')),100),(find_value(loaded,('scheduler_gamma','gamma')),0.5),(find_value(loaded,('eval_interval',)),10))
        update_check(current_index, all(a == b for a,b in expected), str(expected)); current_index += 1
        update_check(current_index, 'adamw' in lower and 'zero_grad' in lower and '.backward(' in compact and 'optimizer.step' in lower, 'AdamW optimization loop'); current_index += 1
        update_check(current_index, 'steplr' in lower and 'step_size' in lower and 'gamma' in lower and 'scheduler.step' in lower, 'StepLR(step_size=100, gamma=0.5)'); current_index += 1
        update_check(current_index, 'teacher_forcing' in lower and 'torch.cat' in lower and any(t in lower for t in ('truth','target')) and 'mse_loss' in lower and find_value(loaded,('output_steps','t_out')) == 10, 'ground-truth feedback and mean 10-step MSE'); current_index += 1
        update_check(current_index, 'autoregressive' in lower and 'torch.cat' in lower and any(t in lower for t in ('prediction','pred')) and ('validate' in lower or 'evaluate' in lower), '10-step predicted feedback'); current_index += 1
        update_check(current_index, find_value(loaded,('eval_interval',)) == 10 and 'for epoch' in lower and ('validate' in lower or 'evaluate' in lower), 'validation every 10 epochs'); current_index += 1
        update_check(current_index, 'relative_l2' in lower and 'best_relative_l2' in lower and any(t in compact for t in ('<best_relative_l2','<=best_relative_l2')), 'save only a lower mean Relative L2'); current_index += 1
        update_check(current_index, 'best_kno_navier_stokes_v1e5.pt' in source, 'best_kno_navier_stokes_v1e5.pt'); current_index += 1
        keys = ('model_state_dict','model_config','data_config','spatial_shape','epoch','best_relative_l2'); update_check(current_index, all(f'"{k}"' in lower or f"\'{k}\'" in lower for k in keys), list(keys)); current_index += 1
        update_check(current_index, 'if__name__=="__main__"' in compact or "if__name__=='__main__'" in compact, 'CLI entrypoint'); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_infer_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['eval:checkpoint','eval:inference_mode','eval:autoregressive_relative_l2']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        path = next(((CANDIDATE_DIR / name).resolve() for name in ('evaluate.py','eval.py') if (CANDIDATE_DIR / name).resolve().is_file()), None)
        if path is None:
            for _ in CHECKPOINT_NAMES:
                update_check(current_index, True, 'validation is integrated into train.py'); current_index += 1
        else:
            lower = path.read_text(encoding='utf-8').lower()
            update_check(current_index, 'load' in lower and 'checkpoint' in lower, str(path)); current_index += 1
            update_check(current_index, '.eval(' in lower and ('no_grad' in lower or 'inference_mode' in lower), str(path)); current_index += 1
            update_check(current_index, 'autoregressive' in lower and 'relative_l2' in lower, str(path)); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_structure(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['model:required_classes','model:op_size_default','model:pointwise_convolutions','model:koopman_matrix','model:fft_evolution','model:six_residual_steps','model:shortcut_tanh','model:no_batch_norm','model:trainable']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        import torch.nn as nn
        path = (CANDIDATE_DIR / config.get('model_file','model.py')).resolve(); spec = importlib.util.spec_from_file_location('kno_candidate_structure', path); module = importlib.util.module_from_spec(spec); sys.modules['kno_candidate_structure'] = module; spec.loader.exec_module(module)
        names = ('EncoderConv2D','DecoderConv2D','KoopmanOperator2D','KNO2D','KNO2DNavierStokes'); classes = {n:getattr(module,n,None) for n in names}; cls = classes['KNO2DNavierStokes']; params = inspect.signature(cls).parameters; values = {'input_channels':10,'in_channels':10,'output_channels':1,'out_channels':1,'spatial_shape':(64,64),'op_size':32,'modes_x':12,'modes_y':12,'evolution_steps':6,'decompose':6,'linear_type':True,'normalization':False}; model = cls(**{k:v for k,v in values.items() if k in params}).cpu(); modules = list(model.modules()); conv = [m for m in modules if isinstance(m,nn.Conv2d)]; bn = [m for m in modules if isinstance(m,nn.modules.batchnorm._BatchNorm)]; matrix = next((p for p in model.parameters() if p.ndim == 4 and tuple(p.shape) == (32,32,12,12) and torch.is_complex(p)),None); source = inspect.getsource(module); compact = ''.join(source.lower().split()); steps = next((int(getattr(m,n)) for m in modules for n in ('evolution_steps','decompose') if getattr(m,n,None) is not None), None)
        update_check(current_index, all(inspect.isclass(v) and issubclass(v,nn.Module) for v in classes.values()), str([n for n,v in classes.items() if not inspect.isclass(v)])); current_index += 1
        update_check(current_index, params.get('op_size') is not None and params['op_size'].default == 32, str(params.get('op_size'))); current_index += 1
        update_check(current_index, len(conv) >= 3 and all(m.kernel_size == (1,1) for m in conv), str([m.kernel_size for m in conv])); current_index += 1
        update_check(current_index, matrix is not None, str(tuple(matrix.shape) if matrix is not None else None)); current_index += 1
        update_check(current_index, 'rfft2' in compact and 'irfft2' in compact and ('einsum("btxy,tfxy->bfxy"' in compact or "einsum('btxy,tfxy->bfxy'" in compact), 'rfft2/einsum/irfft2'); current_index += 1
        update_check(current_index, steps == 6 and any(t in compact for t in ('x=x+x1','x+=x1','latent=latent+self.','latent+=self.','x=x+self.')), f'evolution_steps={steps}'); current_index += 1
        update_check(current_index, 'tanh' in compact and any(t in compact for t in ('shortcut','w0','skip')), '1x1 shortcut followed by tanh'); current_index += 1
        update_check(current_index, not bn, f'batch_norm_count={len(bn)}'); current_index += 1
        update_check(current_index, sum(p.numel() for p in model.parameters()) > 0, 'trainable parameters'); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_executable(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['forward:return_tensor','forward:shape','forward:finite','forward:six_evolutions','forward:pos_compatibility_only']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        path = (CANDIDATE_DIR / config.get('model_file','model.py')).resolve(); spec = importlib.util.spec_from_file_location('kno_candidate_forward',path); module = importlib.util.module_from_spec(spec); sys.modules['kno_candidate_forward'] = module; spec.loader.exec_module(module); cls = getattr(module,'KNO2DNavierStokes'); vals = {'input_channels':10,'output_channels':1,'spatial_shape':(64,64),'op_size':32,'modes_x':12,'modes_y':12,'evolution_steps':6,'decompose':6,'linear_type':True,'normalization':False}; params = inspect.signature(cls).parameters; model = cls(**{k:v for k,v in vals.items() if k in params}).cpu().eval(); gen = torch.Generator(device='cpu').manual_seed(20260824); pos = torch.randn((1,4096,2),generator=gen); fx = torch.randn((1,4096,10),generator=gen); calls = 0
        def hook(_module, _inputs, _output):
            nonlocal calls
            calls += 1
        operator = getattr(module,'KoopmanOperator2D'); handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m,operator)]
        try:
            with torch.no_grad(): output = model(pos,fx)
        finally:
            for handle in handles: handle.remove()
        with torch.no_grad(): shifted = model(pos + 7.0,fx)
        update_check(current_index, isinstance(output,torch.Tensor), type(output).__name__); current_index += 1
        update_check(current_index, isinstance(output,torch.Tensor) and tuple(output.shape) == (1,4096,1), str(tuple(output.shape) if isinstance(output,torch.Tensor) else None)); current_index += 1
        update_check(current_index, isinstance(output,torch.Tensor) and bool(torch.isfinite(output).all()), 'all values finite'); current_index += 1
        update_check(current_index, calls == 6, f'KoopmanOperator2D calls={calls}'); current_index += 1
        update_check(current_index, isinstance(output,torch.Tensor) and bool(torch.equal(output,shifted)), 'changing pos values must not change output'); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_model_forward_equivalence(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['equivalence:state_schema','equivalence:forward_values']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        def load(name,path):
            spec = importlib.util.spec_from_file_location(name,path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
        generated = load('kno_candidate_equivalence',(CANDIDATE_DIR / config.get('model_file','model.py')).resolve()); reference = load('kno_reference_equivalence',(REFERENCE_DIR / config.get('model_file','model.py')).resolve())
        def build(mod):
            cls = getattr(mod,'KNO2DNavierStokes'); vals = {'input_channels':10,'output_channels':1,'spatial_shape':(24,24),'op_size':32,'modes_x':12,'modes_y':12,'evolution_steps':6,'decompose':6,'linear_type':True,'normalization':False}; params = inspect.signature(cls).parameters; return cls(**{k:v for k,v in vals.items() if k in params}).cpu().eval()
        candidate, standard = build(generated), build(reference); gs, rs = list(candidate.state_dict().items()), list(standard.state_dict().items()); same = len(gs) == len(rs) and all(a.shape == b.shape for (_,a),(_,b) in zip(gs,rs)); update_check(current_index, same, f'generated={len(gs)} reference={len(rs)}'); current_index += 1
        if same:
            candidate.load_state_dict({gn:rv for (gn,_),(_,rv) in zip(gs,rs)},strict=True); gen = torch.Generator(device='cpu').manual_seed(20260824); pos = torch.randn((1,576,2),generator=gen); fx = torch.randn((1,576,10),generator=gen)
            with torch.no_grad(): actual, expected = candidate(pos,fx), standard(pos,fx)
            ok = tuple(actual.shape) == tuple(expected.shape) and bool(torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)); detail = f'max_abs_diff={float((actual-expected).abs().max()):.3e}'
        else: ok, detail = False, 'parameter order/shape differs from standard implementation'
        update_check(current_index, ok, detail); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_parameter_update(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['training:teacher_forcing_shape','training:finite_mse','training:finite_gradients','training:parameter_update']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        import torch
        import torch.nn.functional as F
        path = (CANDIDATE_DIR / config.get('model_file','model.py')).resolve(); spec = importlib.util.spec_from_file_location('kno_candidate_training',path); module = importlib.util.module_from_spec(spec); sys.modules['kno_candidate_training'] = module; spec.loader.exec_module(module); cls = getattr(module,'KNO2DNavierStokes'); vals = {'input_channels':10,'output_channels':1,'spatial_shape':(24,24),'op_size':32,'modes_x':12,'modes_y':12,'evolution_steps':6,'decompose':6,'linear_type':True,'normalization':False}; params = inspect.signature(cls).parameters; model = cls(**{k:v for k,v in vals.items() if k in params}).cpu().train(); parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}; before = {n:p.detach().clone() for n,p in parameters.items()}; optimizer = torch.optim.AdamW(parameters.values(),lr=1e-3,weight_decay=1e-5); gen = torch.Generator(device='cpu').manual_seed(20260824); pos = torch.randn((1,576,2),generator=gen); state = torch.randn((1,576,10),generator=gen); target = torch.randn((1,576,10),generator=torch.Generator(device='cpu').manual_seed(20260825)); optimizer.zero_grad(set_to_none=True); losses = []; predictions = []
        for step in range(10):
            prediction = model(pos,state); truth = target[...,step:step+1]; predictions.append(prediction); losses.append(F.mse_loss(prediction,truth)); state = torch.cat((state[...,1:],truth),dim=-1)
        prediction = torch.cat(predictions,dim=-1); loss = torch.stack(losses).mean(); loss.backward(); finite = [n for n,p in parameters.items() if p.grad is not None and bool(torch.isfinite(p.grad).all())]; optimizer.step(); changed = [n for n,p in parameters.items() if not torch.equal(before[n],p.detach())]
        update_check(current_index, tuple(prediction.shape) == tuple(target.shape), f'actual={tuple(prediction.shape)} expected={tuple(target.shape)}'); current_index += 1
        update_check(current_index, bool(torch.isfinite(loss)), str(float(loss.detach()))); current_index += 1
        update_check(current_index, len(finite) == len(parameters), f'{len(finite)}/{len(parameters)}'); current_index += 1
        update_check(current_index, bool(changed), f'changed={len(changed)}/{len(parameters)}'); current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks

def check_smoke_artifacts(config: Dict, caller_globals=None, caller_locals=None) -> list[tuple]:
    """Checkpoint implementation."""
    CHECKPOINT_NAMES = ['artifacts:declarations','artifacts:materialized_optional']
    checks = [(name, False, 'Not executed due to earlier failure') for name in CHECKPOINT_NAMES]
    def update_check(index: int, passed: bool, msg: str):
        checks[index] = (CHECKPOINT_NAMES[index], passed, msg)
    current_index = 0
    try:
        source = '\n'.join((CANDIDATE_DIR / name).resolve().read_text(encoding='utf-8') for name in ('train.py','utils.py','config.yaml') if (CANDIDATE_DIR / name).resolve().is_file()).lower(); checkpoint_name = 'best_kno_navier_stokes_v1e5.pt'; keys = {'model_state_dict','model_config','data_config','spatial_shape','epoch','best_relative_l2'}
        update_check(current_index, checkpoint_name in source and 'train.log' in source and 'metrics' in source, 'checkpoint, training log, and metrics'); current_index += 1
        paths = list(CANDIDATE_DIR.glob('**/' + checkpoint_name))
        if not paths: update_check(current_index, True, 'full-training artifacts are optional during code evaluation')
        else:
            import torch
            try: value = torch.load(paths[0],map_location='cpu',weights_only=False)
            except TypeError: value = torch.load(paths[0],map_location='cpu')
            missing = sorted(keys - set(value)) if isinstance(value,Mapping) else sorted(keys); logs = any(CANDIDATE_DIR.glob('**/train.log')); metrics = any(CANDIDATE_DIR.glob('**/metrics.jsonl')) or any(CANDIDATE_DIR.glob('**/metrics.csv')); update_check(current_index, isinstance(value,Mapping) and not missing and logs and metrics, f'checkpoint={paths[0]}, missing_keys={missing}, log={logs}, metrics={metrics}')
        current_index += 1
    except Exception:
        update_check(current_index, False, traceback.format_exc())
    return checks
