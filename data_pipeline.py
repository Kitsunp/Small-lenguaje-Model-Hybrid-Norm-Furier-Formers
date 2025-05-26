# data_pipeline.py
# Pipeline completo de datos: limpieza, tokenización, WebDataset y DataModule

# Imports de la biblioteca estándar
import json
import tarfile
import io
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Imports de terceros
import numpy as np
from datasets import load_dataset
from datasets import Dataset as HFDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import webdataset as wds
from sklearn.model_selection import StratifiedShuffleSplit

############################################
# MÓDULOS DE DATOS: PROCESAMIENTO Y LIMPIEZA
############################################

class TextCleaningProcessor:
    """
    Clase encargada de la limpieza y procesamiento de los datos textuales.
    Maneja la carga, limpieza, shuffle y estratificación de los datos.
    """
    def __init__(self, max_examples: int, fineweb_ratio: float = 0.9, 
                 val_split_ratio: float = 0.1, seed: int = 42):
        self.max_examples = max_examples
        self.fineweb_ratio = fineweb_ratio
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        
        # Calcular cuántos ejemplos cargar de cada fuente basado en el ratio
        self.max_fineweb_examples = int(self.max_examples * self.fineweb_ratio)
        self.max_math_examples = self.max_examples - self.max_fineweb_examples
        
        # Información sobre el dataset
        self.dataset_info = {
            "source": "HuggingFaceFW/fineweb-edu",
            "additional_source": "HuggingFaceTB/finemath",
            "original_license": "ODC-By v1.0",
            "terms_of_use": "CommonCrawl Terms of Use",
            "modifications": "Text cleaning applied, removed technical elements",
            "mix_ratio": f"{self.fineweb_ratio*100:.0f}% FineWeb, {(1-self.fineweb_ratio)*100:.0f}% FineMath"
        }
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando URLs y elementos técnicos."""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'http[s]?://(?:[a-zA-Z0-9./?=&_-]+)', '', text)
        text = re.sub(r's3://commoncrawl/crawl-data/.*\n?', '', text)
        text = re.sub(r'CC-MAIN-\d{4}-\d{2}', '', text)
        text = re.sub(r'<urn:uuid:[a-f0-9-]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_math_text(self, text: str) -> str:
        """Limpia el texto del dataset de matemáticas."""
        if text is None:
            return ""
            
        # Eliminar URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminar caracteres extraños y normalizar espacio en blanco
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar referencias como "www.ejemplo.com"
        text = re.sub(r'www\.\S+\.\w+', '', text)
        
        # Corregir espacios antes de signos de puntuación
        text = re.sub(r'\s([.,;:!?])', r'\1', text)
        
        # Eliminar referencias a archivos o rutas
        text = re.sub(r'[a-zA-Z0-9_\-\.]+\.(pdf|cgi)', '', text)
        
        # Eliminar paréntesis vacíos o con un solo carácter
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\(\s*.\s*\)', '', text)
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_streaming_data(self, clean_data_dir: Path) -> Path:
        """
        Procesa datos de FineWeb y FineMath usando streaming.
        Limpia los textos y guarda los resultados en la carpeta clean_data.
        """
        clean_data_file = clean_data_dir / "cleaned_texts.jsonl"
        
        # Si ya existe el archivo de datos limpios, saltamos este paso
        if clean_data_file.exists():
            print(f"Archivo de datos limpios encontrado: {clean_data_file}")
            return clean_data_file
            
        print(f"Procesando y limpiando datos (objetivo: {self.max_examples} ejemplos)...")
        
        # Cargar FineWeb con streaming
        fineweb_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            split="train", 
            streaming=True
        )
        
        # Cargar FineMath con streaming
        finemath_stream = load_dataset(
            "HuggingFaceTB/finemath", 
            "finemath-4plus", 
            split="train", 
            streaming=True
        )
        
        # Contador de ejemplos procesados y limpios
        fineweb_count = 0
        finemath_count = 0
        
        # Crear archivo JSONL para guardar textos limpios
        with open(clean_data_file, 'w', encoding='utf-8') as f:
            # Procesar FineWeb primero
            print(f"Procesando FineWeb (objetivo: {self.max_fineweb_examples} ejemplos)...")
            for i, example in enumerate(tqdm(fineweb_stream)):
                if fineweb_count >= self.max_fineweb_examples:
                    break
                    
                text = example.get("text", "")
                cleaned_text = self.clean_text(text)
                
                # Filtrar textos muy cortos o vacíos
                if cleaned_text and len(cleaned_text.split()) > 10:
                    example_data = {
                        "text": cleaned_text,
                        "domain": "fineweb",
                        "id": f"fineweb_{fineweb_count}"
                    }
                    f.write(json.dumps(example_data) + '\n')
                    fineweb_count += 1
                    
                if (i + 1) % 10000 == 0:
                    print(f"Procesados {i+1} ejemplos de FineWeb, limpios: {fineweb_count}")
            
            # Procesar FineMath 
            print(f"Procesando FineMath (objetivo: {self.max_math_examples} ejemplos)...")
            for i, example in enumerate(tqdm(finemath_stream)):
                if finemath_count >= self.max_math_examples:
                    break
                    
                text = example.get("text", "")
                cleaned_text = self.clean_math_text(text)
                
                # Filtrar textos muy cortos o vacíos
                if cleaned_text and len(cleaned_text.split()) > 10:
                    example_data = {
                        "text": cleaned_text,
                        "domain": "math",
                        "id": f"math_{finemath_count}"
                    }
                    f.write(json.dumps(example_data) + '\n')
                    finemath_count += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"Procesados {i+1} ejemplos de FineMath, limpios: {finemath_count}")
        
        print(f"Procesamiento completado:")
        print(f"  - Ejemplos de FineWeb limpios: {fineweb_count}")
        print(f"  - Ejemplos de FineMath limpios: {finemath_count}")
        print(f"  - Total: {fineweb_count + finemath_count}")
        
        # Guardar información de atribución
        attribution_info = {
            "source": self.dataset_info["source"],
            "additional_source": self.dataset_info["additional_source"],
            "original_license": self.dataset_info["original_license"],
            "terms_of_use": self.dataset_info["terms_of_use"],
            "modifications": self.dataset_info["modifications"],
            "mix_ratio": f"{fineweb_count/(fineweb_count+finemath_count)*100:.1f}% FineWeb, "
                       f"{finemath_count/(fineweb_count+finemath_count)*100:.1f}% FineMath",
            "processed_examples": fineweb_count + finemath_count,
            "processing_date": str(datetime.now()),
            "min_words_filter": 10
        }
        
        with open(clean_data_dir / 'dataset_attribution.json', 'w') as f:
            json.dump(attribution_info, f, indent=2)
            
        return clean_data_file

    def shuffle_data(self, clean_data_file: Path, shuffled_dir: Path) -> Path:
        """
        Realiza un shuffle de los datos limpios.
        """
        shuffled_file = shuffled_dir / "shuffled_data.jsonl"
        
        # Si ya existe el archivo de datos barajados, saltamos este paso
        if shuffled_file.exists():
            print(f"Archivo de datos barajados encontrado: {shuffled_file}")
            return shuffled_file
            
        print("Realizando shuffle de los datos limpios...")
        
        # Cargar todos los ejemplos en memoria para el shuffle
        examples = []
        with open(clean_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Shuffle usando NumPy con la semilla especificada
        np.random.seed(self.seed)
        indices = np.random.permutation(len(examples))
        shuffled_examples = [examples[i] for i in indices]
        
        # Guardar ejemplos barajados
        with open(shuffled_file, 'w', encoding='utf-8') as f:
            for example in shuffled_examples:
                f.write(json.dumps(example) + '\n')
                
        print(f"Shuffle completado, {len(shuffled_examples)} ejemplos guardados en {shuffled_file}")
        return shuffled_file

    def stratify_data(self, shuffled_file: Path, stratified_dir: Path) -> tuple[Path, Path]:
        """
        Realiza una estratificación de los datos usando sklearn.
        Divide los datos en conjuntos de entrenamiento y validación,
        manteniendo la proporción de dominios.
        """
        train_file = stratified_dir / "train_data.jsonl"
        val_file = stratified_dir / "val_data.jsonl"
        
        # Si ya existen los archivos de datos estratificados, saltamos este paso
        if train_file.exists() and val_file.exists():
            print(f"Archivos de datos estratificados encontrados: {train_file} y {val_file}")
            return train_file, val_file
            
        print("Realizando estratificación de los datos...")
        
        # Cargar todos los ejemplos
        examples = []
        with open(shuffled_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Extraer dominios para estratificación
        domains = [example['domain'] for example in examples]
        
        # Crear división estratificada con sklearn
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.val_split_ratio,
            random_state=self.seed
        )
        
        # Obtener índices de train y val
        train_indices, val_indices = next(splitter.split(examples, domains))
        
        # Dividir los datos según los índices
        train_examples = [examples[i] for i in train_indices]
        val_examples = [examples[i] for i in val_indices]
        
        # Guardar ejemplos de entrenamiento
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
                
        # Guardar ejemplos de validación
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        
        # Calcular estadísticas
        train_fineweb = sum(1 for ex in train_examples if ex['domain'] == 'fineweb')
        train_math = sum(1 for ex in train_examples if ex['domain'] == 'math')
        val_fineweb = sum(1 for ex in val_examples if ex['domain'] == 'fineweb')
        val_math = sum(1 for ex in val_examples if ex['domain'] == 'math')
        
        print(f"Estratificación completada:")
        print(f"Entrenamiento: {len(train_examples)} ejemplos totales")
        print(f"  - FineWeb: {train_fineweb} ({train_fineweb/len(train_examples)*100:.1f}%)")
        print(f"  - FineMath: {train_math} ({train_math/len(train_examples)*100:.1f}%)")
        print(f"Validación: {len(val_examples)} ejemplos totales")
        print(f"  - FineWeb: {val_fineweb} ({val_fineweb/len(val_examples)*100:.1f}%)")
        print(f"  - FineMath: {val_math} ({val_math/len(val_examples)*100:.1f}%)")
        
        return train_file, val_file

    def get_domain_stats(self, stratified_dir: Path) -> dict:
        """
        Analiza y devuelve estadísticas de distribución por dominio.
        Lee directamente desde los archivos de train y val.
        """
        train_file = stratified_dir / "train_data.jsonl"
        val_file = stratified_dir / "val_data.jsonl"
        
        # Verificar que los archivos existan
        if not train_file.exists() or not val_file.exists():
            raise ValueError("Los archivos de datos estratificados no existen.")
            
        # Analizar archivos de train
        train_examples = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                train_examples.append(json.loads(line))
        
        # Analizar archivos de val
        val_examples = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                val_examples.append(json.loads(line))
        
        # Calcular estadísticas
        train_fineweb = sum(1 for ex in train_examples if ex['domain'] == 'fineweb')
        train_math = sum(1 for ex in train_examples if ex['domain'] == 'math')
        train_total = len(train_examples)
        
        val_fineweb = sum(1 for ex in val_examples if ex['domain'] == 'fineweb')
        val_math = sum(1 for ex in val_examples if ex['domain'] == 'math')
        val_total = len(val_examples)
        
        return {
            "train": {
                "fineweb_percent": train_fineweb / train_total * 100 if train_total else 0,
                "math_percent": train_math / train_total * 100 if train_total else 0,
                "fineweb_count": train_fineweb,
                "math_count": train_math
            },
            "validation": {
                "fineweb_percent": val_fineweb / val_total * 100 if val_total else 0,
                "math_percent": val_math / val_total * 100 if val_total else 0,
                "fineweb_count": val_fineweb,
                "math_count": val_math
            }
        }

############################################
# WEBDATASET: CLASES Y UTILIDADES
############################################

class CountedWebDataset(wds.WebDataset):
    def __init__(self, urls, length, **kwargs):
        super().__init__(urls, **kwargs)
        self.length = length
        
    def __len__(self):
        return self.length

class TextTokenizer:
    """
    Clase encargada de la tokenización y creación de WebDatasets.
    Maneja la conversión de texto a tokens y la creación de archivos TAR.
    """
    def __init__(self, tokenizer, max_seq_len: int, stride: int = 16, 
                 shard_size: int = 1000, num_workers: int = 16):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.shard_size = shard_size
        self.num_workers = min(32, num_workers)  # Limitar a un máximo de 32 workers
        self.num_examples = 0  # Inicializar el atributo
    
    # Función estática para ser pickleable
    @staticmethod
    def identity_fn(x):
        """Función de identidad serializable para reemplazar las lambdas."""
        return x
    
    def _save_example_count(self, prefix, count):
        """Guarda el número de ejemplos en un archivo."""
        count_file = Path(prefix).parent / f"{Path(prefix).name}_count.txt"
        with open(count_file, 'w') as f:
            f.write(str(count))
        print(f"Guardado número de ejemplos: {count} en {count_file}")

    def _load_example_count(self, prefix):
        """Carga el número de ejemplos desde un archivo."""
        count_file = Path(prefix).parent / f"{Path(prefix).name}_count.txt"
        if count_file.exists():
            with open(count_file, 'r') as f:
                return int(f.read().strip())
        return None

    def tokenize_and_create_webdataset(self, train_file: Path, val_file: Path, 
                                     webdataset_dir: Path) -> tuple[str, str]:
        """
        Tokeniza los datos y los convierte al formato WebDataset (archivos TAR).
        Utiliza la API oficial de datasets de Hugging Face para la paralelización.
        """
        train_prefix = webdataset_dir / "train"
        val_prefix = webdataset_dir / "val"
        
        # Verificar si ya existen archivos WebDataset
        train_shards = list(webdataset_dir.glob("train-*.tar"))
        val_shards = list(webdataset_dir.glob("val-*.tar"))
        
        if train_shards and val_shards:
            print(f"Archivos WebDataset encontrados: {len(train_shards)} train shards, {len(val_shards)} val shards")
            
            # Cargar count desde archivo
            saved_count = self._load_example_count(train_prefix)
            if saved_count is not None:
                self.num_examples = saved_count
                print(f"Cargado número de ejemplos desde cache: {self.num_examples}")
            else:
                # Estimar conteo basado en shards y tamaño de shard
                self.num_examples = len(train_shards) * self.shard_size
                print(f"Número estimado de ejemplos (no se encontró cache): {self.num_examples}")
                # Guardar para futuras ejecuciones
                self._save_example_count(train_prefix, self.num_examples)
                
            return str(train_prefix), str(val_prefix)
            
        print("Tokenizando datos y creando archivos WebDataset...")
        
        # Procesar archivos de train y val
        train_shards = self._process_file_to_webdataset(train_file, train_prefix)
        # Guardar el contador de ejemplos después de procesar train
        self._save_example_count(train_prefix, self.num_examples)
        train_examples = self.num_examples
        
        val_shards = self._process_file_to_webdataset(val_file, val_prefix)
        # Restaurar el contador para train después de procesar val
        self.num_examples = train_examples
        
        print(f"WebDataset creado:")
        print(f"  - Train: {train_shards} shards con {self.num_examples} ejemplos")
        print(f"  - Val: {val_shards} shards")
        
        return str(train_prefix), str(val_prefix)  # Convertir a string para evitar problemas con Path
    
    def _tokenize_function(self, example):
        """Función para tokenizar un ejemplo"""
        # Tokenizar el texto con soporte para overflowing tokens
        encodings = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=self.stride
        )
        
        # Convertir tensores a listas para serialización
        input_ids_list = [ids.tolist() for ids in encodings["input_ids"]]
        attention_mask_list = [mask.tolist() for mask in encodings["attention_mask"]] if "attention_mask" in encodings else None
        
        # Crear un nuevo ejemplo para cada fragmento tokenizado
        result = {
            "original_id": [example["id"]] * len(input_ids_list),
            "domain": [example["domain"]] * len(input_ids_list),
            "fragment_idx": list(range(len(input_ids_list))),
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list if attention_mask_list else [None] * len(input_ids_list)
        }
        
        return result
    
    def _create_shard(self, examples, output_prefix, shard_idx):
        """Función para crear un shard WebDataset"""
        tar_filename = f"{output_prefix}-{shard_idx:05d}.tar"
        with tarfile.open(tar_filename, "w") as tar:
            for i, example in enumerate(examples):
                fragment_id = f"{shard_idx:05d}_{i:05d}"
                
                # Crear metadatos
                metadata = {
                    "original_id": example["original_id"],
                    "domain": example["domain"],
                    "fragment_idx": example["fragment_idx"]
                }
                json_content = json.dumps(metadata).encode('utf-8')
                
                # Crear contenido de input_ids y attention_mask
                input_ids_content = json.dumps(example["input_ids"]).encode('utf-8')
                attention_mask_content = json.dumps(example["attention_mask"]).encode('utf-8') if example["attention_mask"] is not None else None
                
                # Añadir archivos al TAR
                json_info = tarfile.TarInfo(f"{fragment_id}.json")
                json_info.size = len(json_content)
                tar.addfile(json_info, io.BytesIO(json_content))
                
                input_ids_info = tarfile.TarInfo(f"{fragment_id}.input.json")
                input_ids_info.size = len(input_ids_content)
                tar.addfile(input_ids_info, io.BytesIO(input_ids_content))
                
                if attention_mask_content:
                    attn_info = tarfile.TarInfo(f"{fragment_id}.mask.json")
                    attn_info.size = len(attention_mask_content)
                    tar.addfile(attn_info, io.BytesIO(attention_mask_content))

    def _process_file_to_webdataset(self, input_file, output_prefix):
        """Función para procesar un archivo completo de forma más eficiente en memoria."""
        print(f"Cargando ejemplos de {input_file}")
        dataset = HFDataset.from_json(str(input_file))
        num_original_examples = len(dataset) # Guardar número original
        print(f"Cargados {num_original_examples} ejemplos originales")

        print(f"Tokenizando con {self.num_workers} procesos en paralelo...")
        # El map() de datasets ya muestra una barra de progreso para la tokenización
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=False, # O True si _tokenize_function maneja lotes
            num_proc=self.num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizando datos"
        )
        print("Tokenización completada. Iniciando escritura de shards...")

        shard_buffer = []
        shard_idx = 0
        fragment_count_total = 0
        fragments_in_shard_buffer = 0

        # Estimamos el número total de lotes para la barra de progreso del iterador
        batch_size_iterator = 500 # El tamaño de lote que usas en .iter()
        try:
            total_items_for_iterator = len(tokenized_dataset)
            total_batches = (total_items_for_iterator + batch_size_iterator - 1) // batch_size_iterator
        except TypeError:
            total_batches = None # No podemos determinar el total
            print("[Advertencia] No se pudo determinar el número total de lotes para la barra de progreso.")

        iterator = tokenized_dataset.iter(batch_size=batch_size_iterator)

        # Crear el objeto tqdm manualmente para tener más control
        pbar = tqdm(iterator, total=total_batches, desc="Procesando fragmentos", unit="batch")

        for batch in pbar: # Iterar sobre el objeto tqdm
            batch_fragments = []
            num_examples_in_batch = len(batch['original_id']) # Asumiendo que las claves tienen la misma longitud de lote

            for i in range(num_examples_in_batch):
                # Asegurarse de que accedemos correctamente si _tokenize_function devolvió listas anidadas
                if isinstance(batch["input_ids"][i], list) and isinstance(batch["input_ids"][i][0], list):
                     num_fragments_in_example = len(batch["input_ids"][i])
                     for j in range(num_fragments_in_example):
                         fragment = {
                            "original_id": batch["original_id"][i][j],
                            "domain": batch["domain"][i][j],
                            "fragment_idx": batch["fragment_idx"][i][j],
                            "input_ids": batch["input_ids"][i][j],
                            "attention_mask": batch["attention_mask"][i][j]
                         }
                         batch_fragments.append(fragment)
                else:
                     # Caso donde _tokenize_function ya aplanó o no hubo fragmentación
                     num_fragments_in_example = 1 # Asumir 1 si no es una lista de listas
                     fragment = {
                        "original_id": batch["original_id"][i],
                        "domain": batch["domain"][i],
                        "fragment_idx": batch["fragment_idx"][i], # Puede ser siempre 0 o una lista
                        "input_ids": batch["input_ids"][i],
                        "attention_mask": batch["attention_mask"][i]
                     }
                     # Ajustar índices si son listas [0] en lugar de valores directos
                     if isinstance(fragment["fragment_idx"], list): fragment["fragment_idx"] = fragment["fragment_idx"][0]
                     if isinstance(fragment["original_id"], list): fragment["original_id"] = fragment["original_id"][0]
                     if isinstance(fragment["domain"], list): fragment["domain"] = fragment["domain"][0]

                     batch_fragments.append(fragment)

            # Añadir fragmentos del batch al buffer del shard
            fragments_processed_in_batch = 0
            for fragment in batch_fragments:
                shard_buffer.append(fragment)
                fragments_in_shard_buffer += 1
                fragment_count_total += 1
                fragments_processed_in_batch += 1

                # Si el buffer está lleno, escribir el shard
                if fragments_in_shard_buffer >= self.shard_size:
                    self._create_shard(shard_buffer, output_prefix, shard_idx)
                    shard_idx += 1
                    shard_buffer = []  # Vaciar buffer
                    fragments_in_shard_buffer = 0

            # Actualizar la barra de progreso
            pbar.set_postfix({"Fragments": f"{fragment_count_total:,}"})

        # Cerrar la barra de progreso al finalizar el bucle
        pbar.close()

        # Escribir el último shard si quedan fragmentos en el buffer
        if shard_buffer:
            print(f"Escribiendo último shard {shard_idx} con {len(shard_buffer)} fragmentos...")
            self._create_shard(shard_buffer, output_prefix, shard_idx)
            num_shards = shard_idx + 1
        else:
            num_shards = shard_idx

        print(f"Proceso completado: {fragment_count_total:,} fragmentos generados en {num_shards} shards.") # Añadir formato
        self.num_examples = fragment_count_total

        return num_shards

    def create_webdataset_datapipeline(self, prefix, shuffle_buffer_size: int = 1000, is_train=True):
        """
        Crea un pipeline de datos basado en WebDataset.
        """
        # Construir el patrón de URL para los shards
        prefix_path = Path(prefix)
        shard_files = sorted(list(prefix_path.parent.glob(f"{prefix_path.name}-*.tar")))
        if not shard_files:
            raise FileNotFoundError(f"No se encontraron archivos de shard para el patrón {prefix}-*.tar")
            
        # Crear una lista de URLs absolutas para cada archivo
        urls = [str(f.absolute()) for f in shard_files]
        print(f"Cargando {len(urls)} shards de {prefix}")
        
        # Ajustar el número de nodos en función de los shards disponibles
        # Usar la clase CountedWebDataset definida fuera del método
        dataset = CountedWebDataset(
            urls, 
            length=self.num_examples,
            shardshuffle=len(shard_files) if is_train else 0,
            nodesplitter=None,
            handler=wds.handlers.warn_and_continue,
            empty_check=False  # Añadir este parámetro para evitar el error
        )
        
        # Reducir los tamaños de buffer para limitar el uso de memoria
        if is_train:
            dataset = dataset.shuffle(shuffle_buffer_size // 2)  # Buffer más pequeño
            dataset = dataset.map(self._decode_example)
            dataset = dataset.batched(500, partial=True)
            dataset = dataset.shuffle(1000)
            dataset = dataset.unbatched()
            
            # Usar funciones pickleables
            dataset = dataset.map_dict(
                text=self.identity_fn,
                domain=self.identity_fn,
                original_id=self.identity_fn
            )
        else:
            dataset = dataset.map(self._decode_example)
        
        return dataset

    def _decode_example(self, sample):
        """Función para decodificar y preparar los ejemplos"""
        # Extraer y convertir datos
        metadata = json.loads(sample["json"].decode("utf-8"))
        input_ids = torch.tensor(json.loads(sample["input.json"].decode("utf-8")))
        
        result = {
            "text": input_ids,
            "domain": metadata["domain"],
            "original_id": metadata["original_id"]
        }
        
        # Añadir máscara de atención si existe
        if "mask.json" in sample:
            attention_mask = torch.tensor(json.loads(sample["mask.json"].decode("utf-8")))
            result["attention_mask"] = attention_mask
            
        return result

############################################
# PYTORCH LIGHTNING DATA MODULE
############################################

class FinewebTextDataModule(pl.LightningDataModule):
    """
    Módulo de datos para cargar y procesar textos combinando FineWeb y FineMath.
    Optimizado para procesar textos largos usando WebDataset y estratificación con sklearn.
    
    Implementa el siguiente flujo:
    1. Limpieza de datos (streaming) y guardado
    2. Shuffle y guardado
    3. Estratificación, shuffle y guardado
    4. Tokenización y guardado
    5. Uso de WebDataset para dataloaders
    """
    def __init__(self, tokenizer, max_seq_len: int, batch_size: int, max_examples: int = 10000, 
                 val_split_ratio: float = 0.1, stride: int = 16, 
                 fineweb_ratio: float = 0.9, seed: int = 42, token: str = None,
                 shuffle_buffer_size: int = 1000, shard_size: int = 40000,
                 num_workers: int = 32):
        """
        Inicializa el módulo de datos.
        
        Args:
            tokenizer: Tokenizador a utilizar
            max_seq_len: Longitud máxima de secuencia para tokenización
            batch_size: Tamaño del lote para carga de datos
            max_examples: Número máximo TOTAL de ejemplos (FineWeb + FineMath)
            val_split_ratio: Proporción de ejemplos para validación
            stride: Solapamiento entre fragmentos consecutivos
            fineweb_ratio: Proporción de ejemplos de FineWeb en el dataset final
            seed: Semilla para reproducibilidad
            token: Token de HuggingFace para acceder a datasets protegidos
            shuffle_buffer_size: Tamaño del buffer para shuffle de WebDataset
            shard_size: Número de ejemplos por shard en WebDataset
            num_workers: Número máximo de procesos para tokenización paralela (máximo 32)
        """
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.token = token
        
        # Inicializar los directorios para datos intermedios
        self.base_dir = Path(f"dataset_{max_examples}")
        self.clean_data_dir = self.base_dir / "clean_data"
        self.shuffled_dir = self.base_dir / "shuffled"
        self.stratified_dir = self.base_dir / "stratified"
        self.tokenized_dir = self.base_dir / "tokenized"
        self.webdataset_dir = self.base_dir / "webdataset"
        
        # Crear directorios si no existen
        for directory in [self.base_dir, self.clean_data_dir, self.shuffled_dir, 
                         self.stratified_dir, self.tokenized_dir, self.webdataset_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Inicializar las clases auxiliares
        self.data_cleaner = TextCleaningProcessor(
            max_examples=max_examples,
            fineweb_ratio=fineweb_ratio,
            val_split_ratio=val_split_ratio,
            seed=seed
        )
        
        self.tokenizer_processor = TextTokenizer(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=stride,
            shard_size=shard_size,
            num_workers=num_workers
        )
        
        # Atributos para datasets
        self.train_dataset = None
        self.val_dataset = None
        
        # Valor de padding
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100349

    def setup(self, stage: Optional[str] = None):
        """
        Prepara los datos para entrenamiento y validación.
        Implementa el flujo completo: limpieza, shuffle, estratificación, tokenización, WebDataset.
        """
        print(f"Configurando dataset con máximo {self.data_cleaner.max_examples} ejemplos totales.")
        print(f"Distribución objetivo: {self.data_cleaner.fineweb_ratio*100:.0f}% FineWeb, "
              f"{(1-self.data_cleaner.fineweb_ratio)*100:.0f}% FineMath")
        
        # Paso 1: Procesar y limpiar datos
        clean_data_file = self.data_cleaner.process_streaming_data(self.clean_data_dir)
        
        # Paso 2: Shuffle de datos
        shuffled_file = self.data_cleaner.shuffle_data(clean_data_file, self.shuffled_dir)
        
        # Paso 3: Estratificación
        train_file, val_file = self.data_cleaner.stratify_data(shuffled_file, self.stratified_dir)
        
        # Paso 4: Tokenización y creación de WebDataset
        train_prefix, val_prefix = self.tokenizer_processor.tokenize_and_create_webdataset(
            train_file, val_file, self.webdataset_dir)
        
        # Pre-verificación de archivos para asegurar que todos sean válidos
        self._verify_shards(self.webdataset_dir)
        
        # Paso 5: Crear pipelines de datos con WebDataset
        self.train_dataset = self.tokenizer_processor.create_webdataset_datapipeline(
            train_prefix, self.shuffle_buffer_size, is_train=True)
        self.val_dataset = self.tokenizer_processor.create_webdataset_datapipeline(
            val_prefix, self.shuffle_buffer_size, is_train=False)
        
        print("Configuración de dataset completada.")
    
    def _verify_shards(self, webdataset_dir: Path):
        """Verifica que todos los shards sean archivos TAR válidos."""
        print("Verificando integridad de shards...")
        shard_files = list(webdataset_dir.glob("*.tar"))
        for i, shard in enumerate(shard_files):
            if i % 10 == 0:  # Verificar solo 1 de cada 10 para rapidez
                try:
                    with tarfile.open(shard, "r") as tar:
                        # Simplemente listar los primeros 5 miembros es suficiente para verificar
                        for j, member in enumerate(tar):
                            if j >= 5:
                                break
                except Exception as e:
                    print(f"Error en shard {shard}: {e}")
                    # Si se encontrara un shard corrupto, podríamos eliminarlo o regenerarlo
                    # Por ahora solo reportamos
        print(f"Verificación completada. Encontrados {len(shard_files)} archivos de shards.")

    def train_dataloader(self) -> DataLoader:
        """Dataloader para entrenamiento."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3,
            drop_last=True,
            multiprocessing_context='spawn'
        )

    def val_dataloader(self) -> DataLoader:
        """Dataloader para validación."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context='spawn'
        )
