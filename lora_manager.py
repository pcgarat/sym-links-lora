import os
import sys
import json
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QScrollArea, QGridLayout, QCheckBox, QLineEdit,
                            QGroupBox, QListWidget, QListWidgetItem, QStackedLayout,
                            QComboBox, QTextEdit, QDialog, QProgressBar, QFormLayout,
                            QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QByteArray, QThread, pyqtSlot, QObject
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PIL import Image
import math
from io import BytesIO
import base64
from civitai import CivitaiAPI
import requests

class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log de actualización Civitai")
        self.setMinimumSize(600, 400)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.close_btn)
        self.setLayout(layout)
    def append_log(self, text):
        print(text)  # También mostrar en consola
        self.text_edit.append(text)
        self.text_edit.ensureCursorVisible()
    def enable_close(self, enable=True):
        self.close_btn.setEnabled(enable)

class CivitaiWorker(QObject):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(int, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current, total
    preview_downloaded = pyqtSignal()
    json_updated = pyqtSignal()
    def __init__(self, api_key, lora_folder):
        super().__init__()
        self.api_key = api_key
        self.lora_folder = lora_folder
        self._abort = False
    def abort(self):
        self._abort = True
    @pyqtSlot()
    def run(self):
        from civitai import CivitaiAPI
        api = CivitaiAPI(api_key=self.api_key, log_func=self.log_signal.emit)
        api.set_preview_callback(self.preview_downloaded.emit)
        api.set_json_callback(self.json_updated.emit)
        ok = 0
        fail = 0
        safetensors = []
        for root, dirs, files in os.walk(self.lora_folder):
            for file in files:
                if file.lower().endswith('.safetensors'):
                    safetensors.append((root, file))
        total = len(safetensors)
        processed = 0
        try:
            for root, file in safetensors:
                if self._abort:
                    self.log_signal.emit("Proceso abortado por el usuario.")
                    break
                safetensor_path = os.path.join(root, file)
                self.log_signal.emit(f"Calculando hash para: {safetensor_path}")
                try:
                    file_hash = api.hash_file(safetensor_path)
                    self.log_signal.emit(f"Hash: {file_hash}")
                    model_info = api.get_model_info_by_hash(file_hash)
                    if model_info:
                        self.log_signal.emit(f"Encontrado en Civitai. Descargando archivos...")
                        api.download_model_files(model_info, root, safetensor_path=safetensor_path)
                        self.log_signal.emit(f"✔️ {file} actualizado.")
                        ok += 1
                    else:
                        self.log_signal.emit(f"❌ {file} no encontrado en Civitai.")
                        fail += 1
                except Exception as e:
                    self.log_signal.emit(f"❌ Error con {file}: {e}")
                    fail += 1
                processed += 1
                self.progress.emit(processed, total)
            self.finished.emit(ok, fail)
        except Exception as e:
            self.error.emit(str(e))

class LoraInfoDialog(QDialog):
    def __init__(self, json_path, parent=None, extra_fields=None):
        super().__init__(parent)
        self.setWindowTitle("Información del LORA")
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self.setMinimumSize(900, 700)
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        # Leer JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Campos principales
        base_model = data.get('baseModel', '')
        model = data.get('model') or {}
        model_nsfw = model.get('nsfw', '')
        model_type = model.get('type', '')
        vbox.addWidget(QLabel(f"<b>baseModel:</b> {base_model}"))
        vbox.addWidget(QLabel(f"<b>model.nsfw:</b> {model_nsfw}"))
        # Mostrar trainedWords si no es Checkpoint
        if model_type != 'Checkpoint':
            trained_words = data.get('trainedWords')
            if trained_words:
                if isinstance(trained_words, list):
                    trained_words_str = ', '.join(str(w) for w in trained_words)
                else:
                    trained_words_str = str(trained_words)
                vbox.addWidget(QLabel(f"<b>trainedWords:</b> {trained_words_str}"))
        # Imágenes
        images = data.get('images') or []
        for idx, img in enumerate(images):
            group = QGroupBox()
            group_layout = QHBoxLayout()
            # Miniatura
            img_url = img.get('url', '')
            local_img = None
            base = os.path.splitext(json_path)[0]
            possible_exts = ['.png', '.jpg', '.jpeg', '.webp']
            preview_path = None
            for ext in possible_exts:
                if idx == 0:
                    candidate = base + f".preview{ext}"
                else:
                    candidate = base + f".{idx}.preview{ext}"
                if os.path.exists(candidate):
                    preview_path = candidate
                    break
            if preview_path:
                local_img = preview_path
            pix = None
            if local_img and os.path.exists(local_img):
                pix = QPixmap(local_img)
            elif img_url:
                try:
                    resp = requests.get(img_url, timeout=5)
                    if resp.status_code == 200:
                        img_data = resp.content
                        pix = QPixmap()
                        pix.loadFromData(img_data)
                except Exception:
                    pix = None
            if pix:
                if pix.width() > 256 or pix.height() > 256:
                    pix = pix.scaled(256, 256, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation)
                img_label = QLabel()
                img_label.setPixmap(pix)
                img_label.setFixedSize(pix.width(), pix.height())
                group_layout.addWidget(img_label)
            # Metadatos
            meta = img.get('meta') or {}
            meta_layout = QFormLayout()
            def meta_val(key):
                return str(meta.get(key, ''))
            meta_layout.addRow("sampler:", QLabel(meta_val('sampler')))
            meta_layout.addRow("cfgScale:", QLabel(meta_val('cfgScale')))
            meta_layout.addRow("Schedule type:", QLabel(meta_val('Schedule type')))
            meta_layout.addRow("Distilled CFG Scale:", QLabel(meta_val('Distilled CFG Scale')))
            meta_layout.addRow("Diffusion in Low Bits:", QLabel(meta_val('Diffusion in Low Bits')))
            meta_layout.addRow("seed:", QLabel(str(meta.get('seed', img.get('seed', '')))))
            meta_layout.addRow("Size:", QLabel(str(meta.get('Size', img.get('size', '')))))
            # --- NUEVO: mostrar campos extra si extra_fields está definido ---
            if extra_fields:
                for field in extra_fields:
                    meta_layout.addRow(f"{field}:", QLabel(str(meta.get(field, ''))))
            # Mostrar resources si no es Checkpoint
            if model_type != 'Checkpoint':
                resources = meta.get('resources', [])
                if isinstance(resources, list) and resources:
                    meta_layout.addRow(QLabel('<b>Resources:</b>'))
                    for res in resources:
                        name = str(res.get('name', ''))
                        weight = str(res.get('weight', ''))
                        meta_layout.addRow(QLabel(f'Resource: {name} (weight: {weight})'))
            meta_widget = QWidget()
            meta_widget.setLayout(meta_layout)
            group_layout.addWidget(meta_widget)
            group.setLayout(group_layout)
            vbox.addWidget(group)
        content.setLayout(vbox)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        self.setLayout(layout)

class LoraManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LORA Model Manager")
        self.setMinimumSize(1000, 800)  # Increased window size
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2d2d2d;
                color: #ffffff;
                font-size: 12px;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QCheckBox {
                color: #ffffff;
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0d47a1;
                border: 1px solid #1565c0;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #1565c0;
            }
        """)
        
        # Default paths
        self.default_lora_path = "./lora"
        self.default_output_path = "./selected_loras"
        
        # Zoom: tamaño de los thumbnails
        self.thumbnail_size = 250  # Tamaño inicial de los thumbnails
        
        # Load saved paths (ANTES de crear widgets)
        self.settings_file = "lora_manager_settings.json"
        self.load_settings()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_hbox = QHBoxLayout()
        self.main_hbox.setContentsMargins(0, 0, 0, 0)
        self.main_hbox.setSpacing(0)
        
        # --- NUEVO: Panel lateral izquierdo ---
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setSpacing(15)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        # Controles de directorios y zoom en el sidebar
        self.lora_path_label = QLabel(f"LORA Path: {self.lora_path}")
        self.output_path_label = QLabel(f"Output Path: {self.output_path}")
        self.lora_path_btn = QPushButton("Change LORA Path")
        self.output_path_btn = QPushButton("Change Output Path")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.lora_path_btn.clicked.connect(self.change_lora_path)
        self.output_path_btn.clicked.connect(self.change_output_path)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.sidebar_layout.addWidget(self.lora_path_label)
        self.sidebar_layout.addWidget(self.lora_path_btn)
        self.sidebar_layout.addWidget(self.output_path_label)
        self.sidebar_layout.addWidget(self.output_path_btn)
        self.sidebar_layout.addWidget(self.zoom_out_btn)
        self.sidebar_layout.addWidget(self.zoom_in_btn)
        # --- NUEVO: API key y botón de actualización ---
        self.civitai_api_key_label = QLabel("Civitai API Key:")
        self.civitai_api_key_input = QLineEdit()
        self.civitai_api_key_input.setPlaceholderText("Introduce tu API key de Civitai...")
        self.civitai_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.civitai_api_key_input.setText(getattr(self, 'civitai_api_key', ''))
        self.civitai_api_key_input.textChanged.connect(self.on_civitai_api_key_changed)
        self.sidebar_layout.addWidget(self.civitai_api_key_label)
        self.sidebar_layout.addWidget(self.civitai_api_key_input)
        self.civitai_update_btn = QPushButton("Actualizar metadatos desde Civitai")
        self.civitai_update_btn.clicked.connect(self.on_civitai_update_clicked)
        self.sidebar_layout.addWidget(self.civitai_update_btn)
        # Barra de progreso para actualización Civitai
        self.civitai_progress = QProgressBar()
        self.civitai_progress.setMinimum(0)
        self.civitai_progress.setMaximum(100)
        self.civitai_progress.setValue(0)
        self.civitai_progress.setTextVisible(True)
        self.civitai_progress.setFormat("0%")
        self.sidebar_layout.addWidget(self.civitai_progress)
        # Panel de resumen de cambios
        self.civitai_summary_group = QGroupBox("Cambios realizados")
        self.civitai_summary_layout = QFormLayout()
        self.civitai_summary_label_previews = QLabel("0")
        self.civitai_summary_label_json = QLabel("0")
        self.civitai_summary_layout.addRow("Previews descargados:", self.civitai_summary_label_previews)
        self.civitai_summary_layout.addRow("JSON actualizados:", self.civitai_summary_label_json)
        self.civitai_summary_group.setLayout(self.civitai_summary_layout)
        self.sidebar_layout.addWidget(self.civitai_summary_group)
        # --- FIN NUEVO ---
        self.sidebar_layout.addStretch(1)
        # Botón para mostrar/ocultar el sidebar
        self.toggle_sidebar_btn = QPushButton("☰")
        self.toggle_sidebar_btn.setFixedWidth(32)
        self.toggle_sidebar_btn.setStyleSheet("font-size: 18px; font-weight: bold; background: #222; color: #fff; border-radius: 6px;")
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        # --- FIN NUEVO ---
        # --- NUEVO: Layout principal horizontal ---
        self.sidebar_container = QVBoxLayout()
        self.sidebar_container.setContentsMargins(0, 0, 0, 0)
        self.sidebar_container.setSpacing(0)
        self.sidebar_container.addWidget(self.toggle_sidebar_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.sidebar_container.addWidget(self.sidebar)
        self.main_hbox.addLayout(self.sidebar_container)
        # Widget central para el resto de la UI
        self.central_vbox = QVBoxLayout()
        self.central_vbox.setSpacing(15)
        self.central_vbox.setContentsMargins(20, 20, 20, 20)
        # Search box
        search_group = QGroupBox("")
        search_layout = QHBoxLayout()
        self.search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by name or path...")
        self.search_box.textChanged.connect(self.filter_loras)
        # Combo para subcarpetas y breadcrumb
        self.folder_breadcrumb = QLabel("")
        self.folder_combo = QComboBox()
        self.folder_combo.setMinimumWidth(120)
        self.folder_combo.currentIndexChanged.connect(self.on_folder_combo_changed)
        search_layout.addWidget(self.search_label)
        search_layout.addWidget(self.search_box)
        search_layout.addWidget(self.folder_breadcrumb)
        search_layout.addWidget(self.folder_combo)
        search_group.setLayout(search_layout)
        self.central_vbox.addWidget(search_group)
        # Available LORAs section
        available_group = QGroupBox("Available LORAs")
        available_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.thumbnail_scroll = scroll
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(20)
        scroll.setWidget(self.thumbnail_widget)
        available_layout.addWidget(scroll)
        available_group.setLayout(available_layout)
        self.central_vbox.addWidget(available_group)
        # Apply button
        self.apply_btn = QPushButton("Apply Selection")
        self.apply_btn.clicked.connect(self.apply_selection)
        self.central_vbox.addWidget(self.apply_btn)
        # Selected LORAs section
        selected_group = QGroupBox("Selected LORAs")
        selected_layout = QVBoxLayout()
        selected_scroll = QScrollArea()
        selected_scroll.setWidgetResizable(False)
        selected_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.selected_widget = QWidget()
        self.selected_layout = QHBoxLayout(self.selected_widget)
        self.selected_layout.setSpacing(20)
        self.selected_widget.setMinimumHeight(self.thumbnail_size + 24)
        selected_scroll.setWidget(self.selected_widget)
        selected_layout.addWidget(selected_scroll)
        # Buttons for managing selected LORAs
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        self.remove_all_btn = QPushButton("Remove All")
        self.refresh_btn = QPushButton("Refresh List")
        self.remove_all_btn.clicked.connect(self.remove_selected_or_all_loras)
        self.refresh_btn.clicked.connect(self.refresh_selected_list)
        buttons_layout.addWidget(self.remove_all_btn)
        buttons_layout.addWidget(self.refresh_btn)
        selected_layout.addLayout(buttons_layout)
        selected_group.setLayout(selected_layout)
        self.central_vbox.addWidget(selected_group)
        # Añadir el central_vbox al main_hbox
        self.main_hbox.addLayout(self.central_vbox)
        # Establecer el layout principal
        main_widget.setLayout(self.main_hbox)
        # Sidebar: restaurar estado guardado
        self.set_sidebar_visible(self.sidebar_visible)
        
        # Dictionary to store selected LORAs
        self.selected_applied_loras = set()
        # Dictionary to store all LORA widgets
        self.all_thumbnail_widgets = {}
        
        # Load LORAs and refresh selected list
        QTimer.singleShot(0, self.load_loras)
        QTimer.singleShot(0, self.refresh_selected_list)
        
        # --- Restaurar estado tras crear widgets y layouts ---
        self.set_sidebar_visible(self.sidebar_visible)
        self.update_folder_combo()
        self.load_loras()
        # Iniciar siempre maximizada
        QTimer.singleShot(0, self.showMaximized)
    
    def refresh_selected_list(self):
        """Refresh the list of selected LORAs"""
        # Clear existing thumbnails
        while self.selected_layout.count():
            item = self.selected_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if not os.path.exists(self.output_path):
            self.selected_widget.setMinimumWidth(0)
            self.selected_widget.setMinimumHeight(self.thumbnail_size + 24)
            return
        # Group files by LORA
        lora_files = {}
        for file in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file)
            # Ahora simplemente agrupa por nombre base, ya que son archivos copiados
            if file.lower().endswith('.safetensors'):
                lora_name = os.path.splitext(file)[0]
                lora_dir = self.output_path
                if lora_name not in lora_files:
                    lora_files[lora_name] = {
                        'model': file_path,
                        'preview': None,
                        'config': None
                    }
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                lora_name = os.path.splitext(file)[0].replace('.preview', '')
                if lora_name in lora_files:
                    lora_files[lora_name]['preview'] = file_path
            elif file.lower().endswith(('.yaml', '.yml', '.json')):
                lora_name = os.path.splitext(file)[0]
                if lora_name in lora_files:
                    lora_files[lora_name]['config'] = file_path
        self.selected_applied_loras.clear()  # Limpiar selección al refrescar panel de abajo
        lora_items = list(lora_files.items())
        for lora_name, files in lora_items:
            lora_dir = os.path.dirname(files['model'])
            preview_path = None
            preview_base = f"{lora_name}.preview"
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(lora_dir, f"{preview_base}{ext}")
                if os.path.exists(test_path):
                    preview_path = test_path
                    break
            if preview_path is None:
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(lora_dir, f"{lora_name}{ext}")
                    if os.path.exists(test_path):
                        preview_path = test_path
                        break
            if preview_path is None:
                preview_path = os.path.join(lora_dir, f"{lora_name}.preview.png")
                if not os.path.exists(preview_path):
                    preview_path = os.path.join(lora_dir, "preview.png")
            thumbnail_widget = self.create_thumbnail_widget(files['model'], lora_name, preview_path, is_applied=True)
            self.selected_layout.addWidget(thumbnail_widget)
        # Ajustar el ancho mínimo del widget contenedor
        num_loras = len(lora_items)
        if num_loras > 0:
            total_width = num_loras * (self.thumbnail_size + self.selected_layout.spacing())
            self.selected_widget.setMinimumWidth(total_width)
        else:
            self.selected_widget.setMinimumWidth(0)
        self.selected_widget.setMinimumHeight(self.thumbnail_size + 24)
        self.update_remove_all_btn_text()
    
    def update_remove_all_btn_text(self):
        if self.selected_applied_loras:
            self.remove_all_btn.setText("Remove Selected")
        else:
            self.remove_all_btn.setText("Remove All")
    
    def remove_selected_or_all_loras(self):
        if self.selected_applied_loras:
            # Eliminar solo los seleccionados
            for lora_path in list(self.selected_applied_loras):
                self.remove_lora(lora_path)
            self.selected_applied_loras.clear()
        else:
            # Eliminar todos
            for file in os.listdir(self.output_path):
                file_path = os.path.join(self.output_path, file)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
        self.refresh_selected_list()
        self.selected_applied_loras.clear()
        self.load_loras()
    
    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.lora_path = settings.get('lora_path', self.default_lora_path)
                self.output_path = settings.get('output_path', self.default_output_path)
                self.thumbnail_size = settings.get('thumbnail_size', 250)
                self.sidebar_visible = settings.get('sidebar_visible', True)
                self.selected_lora_subfolder = settings.get('selected_lora_subfolder', "")
                self.civitai_api_key = settings.get('civitai_api_key', '')
        except FileNotFoundError:
            self.lora_path = self.default_lora_path
            self.output_path = self.default_output_path
            self.thumbnail_size = 250
            self.sidebar_visible = True
            self.selected_lora_subfolder = ""
            self.civitai_api_key = ''
    
    def save_settings(self):
        settings = {
            'lora_path': self.lora_path,
            'output_path': self.output_path,
            'thumbnail_size': self.thumbnail_size,
            'sidebar_visible': self.sidebar_visible,
            'selected_lora_subfolder': self.selected_lora_subfolder,
            'civitai_api_key': getattr(self, 'civitai_api_key', '')
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)
    
    def change_lora_path(self):
        new_path = QFileDialog.getExistingDirectory(self, "Select LORA Directory", self.lora_path)
        if new_path:
            self.lora_path = new_path
            self.lora_path_label.setText(f"LORA Path: {self.lora_path}")
            self.save_settings()
            self.update_folder_combo()
            self.load_loras()
    
    def change_output_path(self):
        new_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_path)
        if new_path:
            self.output_path = new_path
            self.output_path_label.setText(f"Output Path: {self.output_path}")
            self.save_settings()
            self.refresh_selected_list()
    
    def create_gray_placeholder(self, size=(200, 200)):
        """Create a gray placeholder image"""
        image = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        image.fill(QColor(200, 200, 200))  # Gray color
        return QPixmap.fromImage(image)
    
    def create_thumbnail_widget(self, lora_path, preview_name, preview_path, is_applied=False):
        """Create a thumbnail widget for a LORA"""
        thumbnail_widget = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_widget)
        thumbnail_layout.setSpacing(8)  # Add spacing between elements
        # No fijar tamaño, dejar que el layout lo ajuste
        
        # Create a container for the image and its border
        image_container = QWidget()
        image_container.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
            QWidget:hover {
                background-color: #353535;
            }
        """)
        # Ajustar tamaño del contenedor de la imagen
        image_container.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        
        # Make the container clickable
        image_container.setCursor(Qt.CursorShape.PointingHandCursor)
        
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(10, 10, 10, 10)  # Add padding inside the border
        image_layout.setSpacing(5)  # Add spacing between elements
        
        # Create name label
        name_label = QLabel(preview_name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lora_json = os.path.splitext(lora_path)[0] + ".json"
        if os.path.exists(lora_json):
            name_label.setStyleSheet("""
                QLabel {
                    color: #2196f3;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 5px;
                }
            """)
        else:
            name_label.setStyleSheet("""
                QLabel {
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 13px;
                    padding: 5px;
                }
            """)
        name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        image_layout.addWidget(name_label)
        
        # Add relative path
        rel_path = os.path.relpath(os.path.dirname(lora_path), self.lora_path)
        path_label = None
        if rel_path != "." and not is_applied:
            path_label = QLabel(rel_path)
            path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            path_label.setStyleSheet("""
                QLabel {
                    color: #ffffff;
                    font-size: 11px;
                    padding: 2px;
                    background-color: #444857;
                    border-radius: 3px;
                    margin: 2px 5px;
                }
            """)
            path_label.setWordWrap(True)  # Allow text to wrap if too long
            path_label.setCursor(Qt.CursorShape.PointingHandCursor)
            image_layout.addWidget(path_label)
        
        try:
            if os.path.exists(preview_path):
                img = Image.open(preview_path)
                img.thumbnail((self.thumbnail_size, self.thumbnail_size))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue(), "PNG")
            else:
                raise FileNotFoundError
        except Exception:
            # No print, just use placeholder
            pixmap = self.create_gray_placeholder((self.thumbnail_size, self.thumbnail_size))  # Usar tamaño dinámico
        
        # Create image label
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setCursor(Qt.CursorShape.PointingHandCursor)
        image_layout.addWidget(img_label)
        
        # Add the image container to the main layout
        thumbnail_layout.addWidget(image_container)
        
        if not is_applied:
            # Selección por click en el contenedor o imagen, info solo en el título
            def update_selection(selected):
                if selected:
                    image_container.setStyleSheet("""
                        QWidget {
                            background-color: #1b5e20;
                            border: 1px solid #2e7d32;
                            border-radius: 5px;
                        }
                        QWidget:hover {
                            background-color: #2e7d32;
                        }
                    """)
                else:
                    image_container.setStyleSheet("""
                        QWidget {
                            background-color: #2d2d2d;
                            border: 1px solid #3d3d3d;
                            border-radius: 5px;
                        }
                        QWidget:hover {
                            background-color: #353535;
                        }
                    """)
            def handle_select(event):
                current_state = lora_path in self.selected_applied_loras
                new_state = not current_state
                if new_state:
                    self.selected_applied_loras.add(lora_path)
                else:
                    self.selected_applied_loras.discard(lora_path)
                update_selection(new_state)
                self.update_remove_all_btn_text()
            def handle_info(event):
                if os.path.exists(lora_json):
                    try:
                        with open(lora_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        model_type = data.get('model', {}).get('type', '')
                        if model_type == 'Checkpoint':
                            dlg = LoraInfoDialog(lora_json, self)
                            dlg.exec()
                        else:
                            extra_fields = [
                                'prompt', 'steps', 'negativePrompt',
                                'Style Selector Style', 'Style Selector Enabled'
                            ]
                            dlg = LoraInfoDialog(lora_json, self, extra_fields=extra_fields)
                            dlg.exec()
                    except Exception as e:
                        print(f"Error abriendo info LORA: {e}")
            # Asignar eventos
            image_container.mousePressEvent = handle_select
            img_label.mousePressEvent = handle_select
            name_label.mousePressEvent = handle_info
            if path_label is not None:
                path_label.mousePressEvent = handle_select
            update_selection(lora_path in self.selected_applied_loras)
        else:
            # Panel de abajo: selección múltiple por click, sin botón X ni info
            def update_selection(selected):
                if selected:
                    image_container.setStyleSheet("""
                        QWidget {
                            background-color: #1b5e20;
                            border: 1px solid #2e7d32;
                            border-radius: 5px;
                        }
                        QWidget:hover {
                            background-color: #2e7d32;
                        }
                    """)
                else:
                    image_container.setStyleSheet("""
                        QWidget {
                            background-color: #2d2d2d;
                            border: 1px solid #3d3d3d;
                            border-radius: 5px;
                        }
                        QWidget:hover {
                            background-color: #353535;
                        }
                    """)
            def handle_select(event):
                current_state = lora_path in self.selected_applied_loras
                new_state = not current_state
                if new_state:
                    self.selected_applied_loras.add(lora_path)
                else:
                    self.selected_applied_loras.discard(lora_path)
                update_selection(new_state)
                self.update_remove_all_btn_text()
            image_container.mousePressEvent = handle_select
            img_label.mousePressEvent = handle_select
            name_label.mousePressEvent = handle_select
            if path_label is not None:
                path_label.mousePressEvent = handle_select
            update_selection(lora_path in self.selected_applied_loras)
        
        return thumbnail_widget
    
    def load_loras(self):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_layout.count())): 
            self.thumbnail_layout.itemAt(i).widget().setParent(None)
        self.all_thumbnail_widgets.clear()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Walk through la carpeta seleccionada
        row = 0
        col = 0
        container_width = self.thumbnail_scroll.viewport().width()
        self.thumbnail_widget.setFixedWidth(container_width)
        spacing = self.thumbnail_layout.spacing()
        max_cols = max(1, int((container_width + spacing) // (self.thumbnail_size + spacing)))
        # Determinar carpeta a mostrar
        if self.selected_lora_subfolder:
            target_dir = os.path.join(self.lora_path, self.selected_lora_subfolder)
        else:
            target_dir = self.lora_path
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith('.safetensors'):
                    lora_path = os.path.join(root, file)
                    lora_name = os.path.splitext(file)[0]
                    # Find associated files
                    preview_path = None
                    config_path = None
                    # Look for preview image with .preview extension
                    preview_base = f"{lora_name}.preview"
                    for img_file in files:
                        if img_file.lower().startswith(preview_base.lower()) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            preview_path = os.path.join(root, img_file)
                            break
                    # If no .preview image found, try with just the LORA name
                    if preview_path is None:
                        for img_file in files:
                            if img_file.lower().startswith(lora_name.lower()) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                preview_path = os.path.join(root, img_file)
                                break
                    # Look for config file with same base name
                    for config_file in files:
                        config_base = os.path.splitext(config_file)[0]
                        if config_base == lora_name and config_file.lower().endswith(('.yaml', '.yml', '.json')):
                            config_path = os.path.join(root, config_file)
                            break
                    if preview_path is None:
                        preview_path = os.path.join(root, f"{lora_name}.preview.png")
                        if not os.path.exists(preview_path):
                            preview_path = os.path.join(root, "preview.png")
                    # Create thumbnail widget
                    thumbnail_widget = self.create_thumbnail_widget(lora_path, lora_name, preview_path)
                    # Store widget with searchable text
                    search_text = f"{lora_name} {os.path.relpath(root, self.lora_path)}"
                    self.all_thumbnail_widgets[search_text.lower()] = thumbnail_widget
                    # Add to layout
                    self.thumbnail_layout.addWidget(thumbnail_widget, row, col)
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
    
    def filter_loras(self):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_layout.count())): 
            self.thumbnail_layout.itemAt(i).widget().setParent(None)
        
        # Get search text
        search_text = self.search_box.text().lower()
        
        # Filter and display thumbnails
        row = 0
        col = 0
        container_width = self.thumbnail_scroll.viewport().width()
        self.thumbnail_widget.setFixedWidth(container_width)
        spacing = self.thumbnail_layout.spacing()
        max_cols = max(1, int((container_width + spacing) // (self.thumbnail_size + spacing)))
        
        for search_key, widget in self.all_thumbnail_widgets.items():
            if search_text in search_key:
                self.thumbnail_layout.addWidget(widget, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
    
    def apply_selection(self):
        # Create symbolic links for selected LORAs
        for lora_path in self.selected_applied_loras:
            lora_dir = os.path.dirname(lora_path)
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            # Get all associated files
            for file in os.listdir(lora_dir):
                file_path = os.path.join(lora_dir, file)
                if not os.path.isfile(file_path):
                    continue
                # Skip files that are not part of this LORA
                file_base = os.path.splitext(file)[0]
                if not (file_base == lora_name or file.startswith(f"{lora_name}.preview")):
                    continue
                # Create target path with the same filename
                target_path = os.path.join(self.output_path, file)
                # Remove existing link if it exists
                if os.path.exists(target_path):
                    if os.path.islink(target_path):
                        os.unlink(target_path)
                    else:
                        print(f"Warning: {target_path} exists and is not a symbolic link")
                        continue
                # Just before symlink, check again
                if not os.path.exists(file_path):
                    print(f"[DEBUG] Skipping missing file: {file_path}")
                    continue
                else:
                    print(f"[DEBUG] File exists and will be linked: {file_path}")
                    print(f"[DEBUG] Fuente: {file_path}")
                    print(f"    - exists: {os.path.exists(file_path)}")
                    print(f"    - isfile: {os.path.isfile(file_path)}")
                    try:
                        stat = os.stat(file_path)
                        print(f"    - size: {stat.st_size} bytes")
                        print(f"    - permissions: {oct(stat.st_mode)}")
                    except Exception as e:
                        print(f"    - stat error: {e}")
                    print(f"[DEBUG] Destino: {target_path}")
                    print(f"    - exists: {os.path.exists(target_path)}")
                    print(f"    - islink: {os.path.islink(target_path)}")
                    print(f"    - dirname exists: {os.path.exists(os.path.dirname(target_path))}")
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(file_path, target_path)
                    print(f"[DEBUG] Copied file: {file_path} -> {target_path}")
                except Exception as e:
                    import traceback
                    print(f"[DEBUG] Exception copying file for {file}: {e}")
                    traceback.print_exc()
        # Refresh the selected list
        self.refresh_selected_list()
        self.selected_applied_loras.clear()
        self.load_loras()
    
    def remove_lora(self, lora_path):
        """Remove all files for a specific LORA from the output directory"""
        lora_name = os.path.splitext(os.path.basename(lora_path))[0]
        for file in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file)
            file_base = os.path.splitext(file)[0]
            if file_base == lora_name or file.startswith(f"{lora_name}.preview"):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
        self.refresh_selected_list()
    
    def zoom_in(self):
        if self.thumbnail_size < 500:
            self.thumbnail_size += 50
            self.load_loras()
            self.refresh_selected_list()
            self.save_settings()
    
    def zoom_out(self):
        if self.thumbnail_size > 100:
            self.thumbnail_size -= 50
            self.load_loras()
            self.refresh_selected_list()
            self.save_settings()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.load_loras)
        QTimer.singleShot(0, self.refresh_selected_list)

    def toggle_sidebar(self):
        self.set_sidebar_visible(not self.sidebar_visible)
        QTimer.singleShot(0, self.load_loras)

    def update_folder_combo(self):
        # Muestra subcarpetas de la carpeta seleccionada y opción de volver a la carpeta padre
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        # Calcular ruta absoluta de la carpeta actual
        # Fallback robusto: si la carpeta guardada no existe, sube hasta encontrar una válida
        subfolder = getattr(self, 'selected_lora_subfolder', "")
        while True:
            if subfolder:
                current_path = os.path.join(self.lora_path, subfolder)
            else:
                current_path = self.lora_path
            if os.path.isdir(current_path):
                break
            if not subfolder:
                break
            # Subir a la carpeta padre
            subfolder = os.path.dirname(subfolder)
            if subfolder == ".":
                subfolder = ""
        self.selected_lora_subfolder = subfolder
        # Breadcrumb
        rel_current = os.path.relpath(current_path, self.lora_path)
        if rel_current == ".":
            self.folder_breadcrumb.setText("(Root)")
        else:
            self.folder_breadcrumb.setText(rel_current)
        # Opciones del combo: '.. (Parent)' y subcarpetas
        if rel_current != ".":
            self.folder_combo.addItem(".. (Parent)", "..")
        for entry in sorted(os.listdir(current_path)):
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path) and os.listdir(full_path):
                self.folder_combo.addItem(entry, os.path.relpath(full_path, self.lora_path))
        # Selecciona la opción correspondiente a la carpeta guardada
        if rel_current == ".":
            self.folder_combo.setCurrentIndex(-1)
        else:
            found = False
            for i in range(self.folder_combo.count()):
                if self.folder_combo.itemData(i) == rel_current:
                    self.folder_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.folder_combo.setCurrentIndex(-1)
        self.folder_combo.blockSignals(False)
        # Asegura que el sidebar esté en el estado guardado
        self.set_sidebar_visible(self.sidebar_visible)

    def on_folder_combo_changed(self, idx):
        data = self.folder_combo.currentData()
        if data == "..":
            # Subir a la carpeta padre
            if self.selected_lora_subfolder:
                parent = os.path.dirname(self.selected_lora_subfolder)
                if parent == "":
                    self.selected_lora_subfolder = ""
                else:
                    self.selected_lora_subfolder = parent
        elif data:
            self.selected_lora_subfolder = data
        self.save_settings()
        self.update_folder_combo()
        self.load_loras()

    def closeEvent(self, event):
        self.set_sidebar_visible(self.sidebar.isVisible())
        self.save_settings()
        super().closeEvent(event)

    def set_sidebar_visible(self, visible):
        self.sidebar_visible = visible
        self.sidebar.setVisible(visible)
        self.save_settings()

    def on_civitai_api_key_changed(self, text):
        self.civitai_api_key = text
        self.save_settings()

    def on_civitai_update_clicked(self):
        api_key = getattr(self, 'civitai_api_key', '')
        lora_folder = self.lora_path if not self.selected_lora_subfolder else os.path.join(self.lora_path, self.selected_lora_subfolder)
        # Deshabilitar controles
        self.civitai_update_btn.setText("Abortar")
        self.civitai_update_btn.setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;")
        self.civitai_update_btn.setEnabled(True)
        self.civitai_update_btn.clicked.disconnect()
        self.civitai_update_btn.clicked.connect(self.on_civitai_abort_clicked)
        self.civitai_api_key_input.setEnabled(False)
        self.civitai_progress.setValue(0)
        self.civitai_progress.setFormat("0%")
        # Resetear contadores
        self.civitai_summary_label_previews.setText("0")
        self.civitai_summary_label_json.setText("0")
        self.civitai_count_previews = 0
        self.civitai_count_json = 0
        # Crear y mostrar ventana de log
        self.log_dialog = LogDialog(self)
        self.log_dialog.append_log("Iniciando actualización de metadatos desde Civitai...")
        self.log_dialog.show()
        # Lanzar worker en un hilo
        self.worker_thread = QThread()
        self.worker = CivitaiWorker(api_key, lora_folder)
        self.worker.moveToThread(self.worker_thread)
        self.worker.log_signal.connect(self.log_dialog.append_log)
        self.worker.finished.connect(self._on_civitai_update_finished)
        self.worker.error.connect(self._on_civitai_update_error)
        self.worker.progress.connect(self._on_civitai_progress)
        self.worker.preview_downloaded.connect(self._on_civitai_preview_downloaded)
        self.worker.json_updated.connect(self._on_civitai_json_updated)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def on_civitai_abort_clicked(self):
        if hasattr(self, 'worker') and self.worker:
            self.worker.abort()
        self.civitai_update_btn.setEnabled(False)

    def _on_civitai_update_finished(self, ok, fail):
        self.log_dialog.append_log(f"\nActualización completada. {ok} LORAs actualizados, {fail} sin coincidencia.")
        self.log_dialog.enable_close(True)
        self.civitai_update_btn.setText("Actualizar metadatos desde Civitai")
        self.civitai_update_btn.setStyleSheet("")
        self.civitai_update_btn.setEnabled(True)
        self.civitai_update_btn.clicked.disconnect()
        self.civitai_update_btn.clicked.connect(self.on_civitai_update_clicked)
        self.civitai_api_key_input.setEnabled(True)
        self.civitai_progress.setValue(100)
        self.civitai_progress.setFormat("100%")
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.load_loras()
        self.refresh_selected_list()

    def _on_civitai_update_error(self, msg):
        self.log_dialog.append_log(f"\nError: {msg}")
        self.log_dialog.enable_close(True)
        self.civitai_update_btn.setText("Actualizar metadatos desde Civitai")
        self.civitai_update_btn.setStyleSheet("")
        self.civitai_update_btn.setEnabled(True)
        self.civitai_update_btn.clicked.disconnect()
        self.civitai_update_btn.clicked.connect(self.on_civitai_update_clicked)
        self.civitai_api_key_input.setEnabled(True)
        self.civitai_progress.setValue(0)
        self.civitai_progress.setFormat("0%")
        self.worker_thread.quit()
        self.worker_thread.wait()

    def _on_civitai_progress(self, processed, total):
        if total > 0:
            percent = int(processed * 100 / total)
        else:
            percent = 100
        self.civitai_progress.setValue(percent)
        self.civitai_progress.setFormat(f"{percent}%")

    def _on_civitai_preview_downloaded(self):
        self.civitai_count_previews += 1
        self.civitai_summary_label_previews.setText(str(self.civitai_count_previews))

    def _on_civitai_json_updated(self):
        self.civitai_count_json += 1
        self.civitai_summary_label_json.setText(str(self.civitai_count_json))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoraManager()
    window.show()
    sys.exit(app.exec()) 