# training_system.py
# Sistema de entrenamiento: Optimizador Muon, Lightning Module y script principal

# Imports de la biblioteca estándar
import math
import os
import argparse
import sys
import multiprocessing
from typing import Dict, List

# Imports de terceros
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names

# Imports locales
from model_architecture import MultiModalModel
from data_pipeline import FinewebTextDataModule

# Configuración de entorno y de Torch
os.environ["HF_TOKEN"] = "Tu token"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
torch.set_default_dtype(torch.float16)

############################################
# IMPLEMENTACIÓN DEL OPTIMIZADOR MUON
############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Iteración Newton-Schulz para calcular la potencia cero / ortogonalización de G.
    """
    assert len(G.shape) == 2
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = (b * A + c * A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    
    return X

class Muon(torch.optim.Optimizer):
    """
    Optimizador Muon que combina Newton-Schulz para parámetros matriciales 
    con AdamW para parámetros no matriciales.
    
    Args:
        lr: Learning rate base
        wd: Weight decay
        muon_params: Lista de parámetros matriciales para Muon
        momentum: Factor de momentum para Muon
        nesterov: Si usar Nesterov momentum
        ns_steps: Número de pasos Newton-Schulz
        adamw_params: Lista de parámetros para AdamW
        adamw_betas: Betas para AdamW
        adamw_eps: Epsilon para AdamW
    """
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Marcar parámetros para Muon
        if muon_params is not None:
            for p in muon_params:
                assert p.ndim == 2, f"Expected 2D parameter for Muon, got {p.ndim}D"
                self.state[p]["use_muon"] = True
        
        # Marcar parámetros para AdamW
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        """
        Ajusta la tasa de aprendizaje basada en las dimensiones del parámetro.
        """
        A, B = param_shape[:2]
        adjusted_factor = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_factor
        return adjusted_lr

    def step(self, closure=None):
        """Realizar un paso de optimización."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

############################################
# LIGHTNING MODULE: ENTRENAMIENTO DEL MODELO
############################################

class MultiModalLightningModule(pl.LightningModule):
    """
    Módulo de PyTorch Lightning para entrenar el modelo FANformer.
    
    Incluye:
    - Configuración del optimizador Muon
    - Métricas de entrenamiento y validación por dominio
    - Scheduler de learning rate
    """
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 num_heads: int, num_decoder_layers: int, ff_dim: int,
                 lr: float = 1e-4, dropout: float = 0.12,
                 NUM_GQA_GROUPS: int = 4, p: float = 0.15):
        super().__init__()
        self.save_hyperparameters()
        
        # Inicializar tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100349
        except Exception as e:
            print(f"[WARNING] Error al cargar tokenizer: {e}")
            self.pad_token_id = 100349
            tokenizer = None
        
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        # Inicializar modelo
        self.model = MultiModalModel(
            vocab_size, embed_dim, max_seq_len,
            num_heads, num_decoder_layers, ff_dim,
            dropout=dropout, num_gqa_groups=NUM_GQA_GROUPS, p=p
        )
        self.model = self.model.to(torch.bfloat16) 
        self.model.tokenizer = tokenizer
        
        # Compilar modelo
        self.model = torch.compile(self.model, backend="inductor",
                                   mode="max-autotune", fullgraph=True)
        
        # Configuración de entrenamiento
        self.lr = lr
        self.next_token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.validation_step_outputs = []

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Paso de entrenamiento."""
        text = batch["text"].to(self.device)
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        
        # Cálculo de la pérdida
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        
        # Cálculo de accuracy
        predicted_tokens = torch.argmax(next_token_logits, dim=-1)
        mask = next_token_targets != self.pad_token_id
        correct = (predicted_tokens == next_token_targets) & mask
        total_valid = mask.sum().float()
        
        if total_valid > 0:
            accuracy = correct.sum().float() / total_valid
            self.log("train_accuracy", accuracy, prog_bar=True, batch_size=text.size(0))
        
        self.log("train_loss", loss, prog_bar=True, batch_size=text.size(0))
        return loss

    def on_validation_epoch_start(self):
        """Reiniciar acumuladores al inicio de validación."""
        self.validation_step_outputs = []

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Paso de validación con métricas por dominio."""
        text = batch["text"].to(self.device)
        domains = [str(d) for d in batch.get("domain", ["unknown"] * text.size(0))]
        
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        
        # Cálculo de la pérdida
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        
        # Cálculo de accuracy global
        predicted_tokens = torch.argmax(next_token_logits, dim=-1)
        mask = next_token_targets != self.pad_token_id
        correct = (predicted_tokens == next_token_targets) & mask
        total_valid = mask.sum().float()
        
        if total_valid > 0:
            accuracy = correct.sum().float() / total_valid
            self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True, batch_size=text.size(0))
        
        # Guardar resultados por ejemplo para análisis por dominio
        example_results = []
        for i in range(text.size(0)):
            example_correct = correct[i].sum().item()
            example_total = mask[i].sum().item()
            example_domain = domains[i]
            
            example_results.append({
                "domain": example_domain,
                "correct": example_correct,
                "total": example_total
            })
        
        self.validation_step_outputs.extend(example_results)
        self.log("validation_loss", loss, prog_bar=True, sync_dist=True, batch_size=text.size(0))
        return loss

    def on_validation_epoch_end(self):
        """Procesar métricas de validación por dominio."""
        # Recopilar resultados de todos los procesos en entrenamiento distribuido
        if self.trainer.world_size > 1:
            all_results = self.all_gather(self.validation_step_outputs)
            gathered_results = []
            for results_list in all_results:
                gathered_results.extend(results_list)
            self.validation_step_outputs = gathered_results
        
        # Organizar resultados por dominio
        domain_stats = {}
        
        for result in self.validation_step_outputs:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}
            
            domain_stats[domain]["correct"] += result["correct"]
            domain_stats[domain]["total"] += result["total"]
        
        # Calcular y registrar accuracy por dominio
        for domain, stats in domain_stats.items():
            if stats["total"] > 0:
                domain_accuracy = stats["correct"] / stats["total"]
                self.log(f"val_accuracy_{domain}", domain_accuracy, sync_dist=True, batch_size=1)
        
        # Calcular diferencia entre dominios principales
        domains = list(domain_stats.keys())
        if len(domains) >= 2:
            main_domains = [d for d in domains if d in ["fineweb", "math"]]
            if len(main_domains) >= 2 and all(domain_stats[d]["total"] > 0 for d in main_domains):
                acc_diff = abs(
                    domain_stats[main_domains[0]]["correct"] / domain_stats[main_domains[0]]["total"] -
                    domain_stats[main_domains[1]]["correct"] / domain_stats[main_domains[1]]["total"]
                )
                self.log("val_accuracy_domain_diff", acc_diff, sync_dist=True, batch_size=1)
        
        # Limpiar acumuladores
        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        """
        Configura el optimizador Muon híbrido.
        
        - Parámetros matriciales (2D) usan Muon con Newton-Schulz
        - Parámetros no matriciales usan AdamW
        - Weight decay uniforme para todos los parámetros
        """
        model = self.model
        
        # Separar parámetros matriciales (para Muon) de no matriciales (para AdamW)
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name and p.requires_grad
        ]
        
        adamw_params = [
            p for name, p in model.named_parameters() 
            if (p.ndim != 2 or "embed_tokens" in name or "lm_head" in name) and p.requires_grad
        ]
        
        print(f"[OPTIMIZER] Muon parámetros: {len(muon_params)}")
        print(f"[OPTIMIZER] AdamW parámetros: {len(adamw_params)}")
        
        # Configurar el optimizador Muon híbrido
        optimizer = Muon(
            lr=self.lr,
            wd=0.013,  # Weight decay uniforme
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_betas=(0.9, 0.99),
            adamw_eps=1e-8
        )
        
        # Scheduler con warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=64500,
            T_mult=1,
            eta_min=self.lr / 7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_loss",
                "interval": "step",
                "frequency": 1
            }
        }

############################################
# FUNCIÓN DE UTILIDAD PARA INSPECCIÓN
############################################

def inspect_model_parameters(model: torch.nn.Module, model_name: str = "Model"):
    """
    Inspecciona y muestra información detallada sobre los parámetros del modelo.
    
    Args:
        model: Modelo de PyTorch a inspeccionar
        model_name: Nombre del modelo para mostrar en el reporte
    """
    print(f"\n--- Inspecting Parameters for {model_name} ---")
    total_params = 0
    dtype_counts = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            param_dtype = param.dtype

            print(f"  Parameter: {name:<70} | Shape: {str(param.shape):<25} | Dtype: {str(param_dtype):<15} | Num params: {num_params}")

            if param_dtype not in dtype_counts:
                dtype_counts[param_dtype] = 0
            dtype_counts[param_dtype] += num_params

    print(f"\n  Total trainable parameters: {total_params:,}")
    print(f"  Parameter Dtype Distribution:")
    for dtype, count in dtype_counts.items():
        percentage = (count / total_params) * 100 if total_params > 0 else 0
        print(f"    - {str(dtype):<15}: {count:,} params ({percentage:.2f}%)")

    # Inspeccionar buffers
    buffer_dtype_counts = {}
    total_buffer_elements = 0
    print(f"\n  Buffer Dtype Distribution:")
    for name, buffer_tensor in model.named_buffers():
        num_elements = buffer_tensor.numel()
        total_buffer_elements += num_elements
        buffer_dtype = buffer_tensor.dtype
        print(f"  Buffer:    {name:<70} | Shape: {str(buffer_tensor.shape):<25} | Dtype: {str(buffer_dtype):<15} | Num elements: {num_elements}")
        if buffer_dtype not in buffer_dtype_counts:
            buffer_dtype_counts[buffer_dtype] = 0
        buffer_dtype_counts[buffer_dtype] += num_elements

    if total_buffer_elements > 0:
        for dtype, count in buffer_dtype_counts.items():
            percentage = (count / total_buffer_elements) * 100
            print(f"    - {str(dtype):<15}: {count:,} elements ({percentage:.2f}%)")
    else:
        print("    No buffers found or all buffers are empty.")

    print(f"--- End of Inspection for {model_name} ---\n")

############################################
# CONFIGURACIÓN Y FUNCIONES AUXILIARES
############################################

def setup_model_config(args):
    """Configuración del modelo basada en argumentos."""
    return {
        "VOCAB_SIZE": None,  # Se configurará con el tokenizer
        "EMBED_DIM": 768,
        "NUM_HEADS": 12,
        "NUM_DECODER_LAYERS": 12,
        "FF_DIM": 2048, 
        "LR": 6e-4,
        "batch_size": args.batch_size,
        "NUM_GQA_GROUPS": 6,
        "BASE_DROPOUT": 0.08,
        "p": args.p
    }

def setup_mode_config(args):
    """Configuración específica del modo de entrenamiento."""
    return {
        "MAX_SEQ_LEN": 1024,
        "max_examples": 300000
    }

def setup_trainer_config():
    """Configuración del trainer de PyTorch Lightning."""
    return {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "precision": "bf16-true",
        "accumulate_grad_batches": 1,
        "enable_checkpointing": False,
        "val_check_interval": 150,
        "limit_val_batches": 15,
        "gradient_clip_val": 1,
        "gradient_clip_algorithm": "norm",
        "strategy": "ddp_find_unused_parameters_true",
        "enable_progress_bar": True
    }

############################################
# BLOQUE PRINCIPAL: ENTRENAMIENTO E INFERENCIA
############################################

if __name__ == "__main__":
    # Configurar multiprocessing
    multiprocessing.set_start_method(method='spawn', force=True)

    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Entrenamiento o Inferencia del modelo FANformer (preentrenamiento)"
    )
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Modo de operación: train o inference")
    parser.add_argument("--epochs_text", type=int, default=2,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Tamaño del batch")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Longitud máxima a generar (tokens adicionales)")
    parser.add_argument("--min_length", type=int, default=1,
                        help="Longitud mínima a generar antes de considerar EOS")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Valor de top_k para sampling")
    parser.add_argument("--top_p", type=float, default=0.85,
                        help="Valor de top_p para sampling")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Factor de temperatura para escalado de logits")
    parser.add_argument("--p", type=float, default=0.14,
                        help="Proporción de modelado periódico para FANformer")
    parser.add_argument("--stride", type=int, default=16,
                        help="Solapamiento entre fragmentos de texto")
    parser.add_argument("--checkpoint_path", type=str, default="./modelo_final/modelo_final_fanformer1.ckpt",
                        help="Ruta del checkpoint para inferencia")
    
    args = parser.parse_args()

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    
    # Configuraciones
    config = setup_model_config(args)
    config["VOCAB_SIZE"] = tokenizer.vocab_size
    
    mode_config = setup_mode_config(args)
    
    # Limpiar caché de CUDA
    torch.cuda.empty_cache()

    if args.mode == "train":
        print(f"[INFO] Entrenamiento en modo DECODER-ONLY utilizando FineWeb-Edu (sample-10BT)")
        print(f"[CONFIG] Utilizando FANformer con p={args.p}")
        print(f"[CONFIG] Longitud máxima de secuencia: {mode_config['MAX_SEQ_LEN']}")
        print(f"[CONFIG] Tamaño de batch: {config['batch_size']}")
        print(f"[CONFIG] Stride para fragmentos: {args.stride}")
    
        # Inicializar el data module
        data_module = FinewebTextDataModule(
            tokenizer=tokenizer, 
            max_seq_len=mode_config["MAX_SEQ_LEN"],
            batch_size=config["batch_size"],
            max_examples=mode_config["max_examples"],
            stride=args.stride
        )

        # Inicializar el modelo
        model_module = MultiModalLightningModule(
            vocab_size=config["VOCAB_SIZE"],
            embed_dim=config["EMBED_DIM"],
            max_seq_len=mode_config["MAX_SEQ_LEN"],
            num_heads=config["NUM_HEADS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            ff_dim=config["FF_DIM"],
            lr=config["LR"],
            dropout=config["BASE_DROPOUT"],
            NUM_GQA_GROUPS=config["NUM_GQA_GROUPS"],
            p=config["p"]
        )
        
        # Inspección de parámetros
        if hasattr(model_module, 'model') and isinstance(model_module.model, torch.nn.Module):
            inspect_model_parameters(model_module.model, "MultiModalModel")
        else:
            inspect_model_parameters(model_module, "MultiModalLightningModule")

        # Configurar y ejecutar entrenamiento
        trainer_config = setup_trainer_config()
        trainer = pl.Trainer(max_epochs=args.epochs_text, **trainer_config)
        trainer.fit(model_module, datamodule=data_module)
        
        # Guardar el modelo
        os.makedirs('./modelo_final', exist_ok=True)
        final_ckpt = args.checkpoint_path
        trainer.save_checkpoint(final_ckpt)
        print(f"[SUCCESS] Modelo guardado exitosamente en {final_ckpt}")
        
    else:
        # Modo inferencia
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] No se encontró el checkpoint en {checkpoint_path}")
            # Intentar rutas alternativas
            alt_paths = [
                "./modelo_final/fanformer_finetuned_final.ckpt",
                "./modelo_final/modelo_final_fanformer1.ckpt"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"[INFO] Usando checkpoint alternativo: {checkpoint_path}")
                    break
            else:
                print("[ERROR] No se encontró ningún checkpoint válido")
                sys.exit(1)
            
        print(f"[INFO] Modo de inferencia activado utilizando FANformer con p={args.p}")
        
        # Cargar modelo para inferencia
        model_inference = MultiModalLightningModule(
            vocab_size=config["VOCAB_SIZE"],
            embed_dim=config["EMBED_DIM"],
            max_seq_len=mode_config["MAX_SEQ_LEN"],
            num_heads=config["NUM_HEADS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            ff_dim=config["FF_DIM"],
            lr=config["LR"],
            dropout=config["BASE_DROPOUT"],
            NUM_GQA_GROUPS=config["NUM_GQA_GROUPS"],
            p=config["p"]
        )
        
        # Cargar pesos del checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_inference.load_state_dict(ckpt["state_dict"])
        model_inference.eval()
        model_inference.to(torch.bfloat16)
        model_inference.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Bucle de inferencia interactiva
        print("\n=== Modo Inferencia ===")
        print("Ingresa tus prompts o escribe 'salir' para terminar")
        print("="*50)
        
        while True:
            prompt = input("\nPrompt> ")
            if prompt.lower() in ["salir", "exit", "quit"]:
                break
            
            # Tokenizar prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Generar respuesta
            try:
                with torch.no_grad():
                    generated_ids = model_inference.model.forward_inference(
                        input_ids,
                        max_length=args.max_length,
                        min_length=args.min_length,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature
                    )
                
                # Decodificar y mostrar resultado
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print("\n=== Resultado ===")
                print(f"Texto generado: {generated_text}")
                print("="*50)
                
            except Exception as e:
                print(f"[ERROR] Error durante la generación: {e}")
                continue
            
        print("\n¡Gracias por usar el modelo de inferencia!")
