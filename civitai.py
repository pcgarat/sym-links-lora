import os
import hashlib
import requests
import json

CIVITAI_API_URL = "https://civitai.com/api/v1"

class CivitaiAPI:
    def __init__(self, api_key=None, log_func=None):
        self.api_key = api_key
        self.log_func = log_func  # función para logs opcional
        self._preview_callback = None
        self._json_callback = None

    def set_api_key(self, api_key):
        self.api_key = api_key

    def set_log_func(self, log_func):
        self.log_func = log_func

    def set_preview_callback(self, cb):
        self._preview_callback = cb

    def set_json_callback(self, cb):
        self._json_callback = cb

    def get_headers(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def hash_file(self, file_path, chunk_size=1024*1024):
        """Calcula el hash SHA256 de un archivo (safetensors)."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_model_info_by_hash(self, file_hash):
        """Busca información de un modelo en Civitai por hash."""
        url = f"{CIVITAI_API_URL}/model-versions/by-hash/{file_hash}"
        resp = requests.get(url, headers=self.get_headers())
        if resp.status_code == 200:
            return resp.json()
        else:
            return None

    def download_model_files(self, model_info, dest_folder, safetensor_path=None):
        """Descarga todos los archivos de metadatos relevantes del modelo (preview, config, json, yaml, etc), pero NO el safetensor. Además guarda la respuesta completa de la API en un .json con el mismo nombre que el safetensor."""
        if not model_info:
            return False
        os.makedirs(dest_folder, exist_ok=True)
        # Guardar la respuesta de la API en un .json con el mismo nombre que el safetensor
        if safetensor_path:
            base = os.path.splitext(safetensor_path)[0]
            json_path = base + ".json"
            new_json = json.dumps(model_info, ensure_ascii=False, indent=2)
            old_json = None
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    old_json = f.read()
            if old_json != new_json:
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(new_json)
                if self.log_func:
                    self.log_func(f"Guardada respuesta de la API en: {json_path}")
                if self._json_callback:
                    self._json_callback()
        # Descargar archivos de metadatos (NO el safetensor)
        for file in model_info.get("files", []):
            file_url = file.get("downloadUrl")
            file_name = file.get("name")
            if file_url and file_name:
                ext = os.path.splitext(file_name)[1].lower()
                # Descargar todo excepto el .safetensors
                if ext != ".safetensors":
                    dest_path = os.path.join(dest_folder, file_name)
                    # Si es .gguf y ya existe, NO descargar
                    if ext == ".gguf" and os.path.exists(dest_path):
                        if self.log_func:
                            self.log_func(f"Archivo GGUF ya existe, no se descarga: {file_name}")
                        continue
                    need_download = True
                    if os.path.exists(dest_path):
                        # Si ya existe, no lo contamos
                        need_download = False
                    if need_download:
                        if self.log_func:
                            self.log_func(f"Descargando archivo de metadatos: {file_name} → {dest_path}")
                        self._download_file(file_url, dest_path)
                        if self._preview_callback and ext in [".png", ".jpg", ".jpeg", ".webp"]:
                            self._preview_callback()
        # Descargar SIEMPRE la primera imagen del array images como .preview.ext_img
        if safetensor_path and model_info.get("images"):
            first_img = model_info["images"][0]
            img_url = first_img.get("url")
            if img_url:
                ext_img = os.path.splitext(img_url)[1]
                preview_path = os.path.splitext(safetensor_path)[0] + f".preview{ext_img}"
                if not os.path.exists(preview_path):
                    if self.log_func:
                        self.log_func(f"Descargando primer preview principal: {os.path.basename(preview_path)} → {preview_path}")
                    self._download_file(img_url, preview_path)
                    if self._preview_callback:
                        self._preview_callback()
        # Descargar imágenes de preview y renombrarlas
        # Si hay un safetensor o gguf, usar ese nombre base
        base_preview = None
        if safetensor_path:
            base_preview = os.path.splitext(safetensor_path)[0]
        else:
            # Buscar un gguf en la carpeta
            for f in os.listdir(dest_folder):
                if f.lower().endswith('.gguf'):
                    base_preview = os.path.splitext(os.path.join(dest_folder, f))[0]
                    break
        # --- Renombrado incremental para previews ---
        preview_count = 0
        for img in model_info.get("images", []):
            img_url = img.get("url")
            if img_url and base_preview:
                ext = os.path.splitext(img_url)[1]
                if preview_count == 0:
                    preview_name = base_preview + f".preview{ext}"
                else:
                    preview_name = base_preview + f".{preview_count}.preview{ext}"
                dest_path = preview_name
                need_download = True
                if os.path.exists(dest_path):
                    need_download = False
                if need_download:
                    if self.log_func:
                        self.log_func(f"Descargando imagen de preview: {os.path.basename(dest_path)} → {dest_path}")
                    self._download_file(img_url, dest_path)
                    if self._preview_callback:
                        self._preview_callback()
                preview_count += 1
        return True

    def _download_file(self, url, dest_path):
        resp = requests.get(url, headers=self.get_headers(), stream=True)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Error downloading {url}: {resp.status_code}")

    def process_lora_folder(self, lora_folder, recursive=True):
        """Busca todos los safetensors en la carpeta y subcarpetas, busca info en Civitai y descarga/actualiza archivos de metadatos (NO el safetensor)."""
        results = []
        for root, dirs, files in os.walk(lora_folder):
            for file in files:
                if file.lower().endswith('.safetensors'):
                    safetensor_path = os.path.join(root, file)
                    file_hash = self.hash_file(safetensor_path)
                    model_info = self.get_model_info_by_hash(file_hash)
                    if model_info:
                        self.download_model_files(model_info, root, safetensor_path=safetensor_path)
                        results.append((safetensor_path, True))
                    else:
                        results.append((safetensor_path, False))
            if not recursive:
                break
        return results
