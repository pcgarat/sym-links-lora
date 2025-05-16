import os
import sys
import json
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QScrollArea, QGridLayout, QCheckBox, QLineEdit,
                            QGroupBox, QListWidget, QListWidgetItem, QStackedLayout)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PIL import Image
import math
from io import BytesIO

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
        
        # Load saved paths
        self.settings_file = "lora_manager_settings.json"
        self.load_settings()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)  # Add spacing between sections
        layout.setContentsMargins(20, 20, 20, 20)  # Add margins around the layout
        
        # Path selection
        path_group = QGroupBox("Paths")
        path_layout = QHBoxLayout()
        path_layout.setSpacing(10)
        
        self.lora_path_label = QLabel(f"LORA Path: {self.lora_path}")
        self.output_path_label = QLabel(f"Output Path: {self.output_path}")
        
        self.lora_path_btn = QPushButton("Change LORA Path")
        self.output_path_btn = QPushButton("Change Output Path")
        
        # Zoom controls
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        
        self.lora_path_btn.clicked.connect(self.change_lora_path)
        self.output_path_btn.clicked.connect(self.change_output_path)
        
        path_layout.addWidget(self.lora_path_label)
        path_layout.addWidget(self.lora_path_btn)
        path_layout.addWidget(self.output_path_label)
        path_layout.addWidget(self.output_path_btn)
        path_layout.addWidget(self.zoom_out_btn)
        path_layout.addWidget(self.zoom_in_btn)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Search box
        search_group = QGroupBox("Search")
        search_layout = QHBoxLayout()
        self.search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by name or path...")
        self.search_box.textChanged.connect(self.filter_loras)
        search_layout.addWidget(self.search_label)
        search_layout.addWidget(self.search_box)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Available LORAs section
        available_group = QGroupBox("Available LORAs")
        available_layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(20)  # Add spacing between thumbnails
        scroll.setWidget(self.thumbnail_widget)
        available_layout.addWidget(scroll)
        
        available_group.setLayout(available_layout)
        layout.addWidget(available_group)
        
        # Apply button
        self.apply_btn = QPushButton("Apply Selection")
        self.apply_btn.clicked.connect(self.apply_selection)
        layout.addWidget(self.apply_btn)
        
        # Selected LORAs section
        selected_group = QGroupBox("Selected LORAs")
        selected_layout = QVBoxLayout()
        
        selected_scroll = QScrollArea()
        selected_scroll.setWidgetResizable(True)
        self.selected_widget = QWidget()
        self.selected_layout = QGridLayout(self.selected_widget)
        self.selected_layout.setSpacing(20)  # Add spacing between thumbnails
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
        layout.addWidget(selected_group)
        
        # Dictionary to store selected LORAs
        self.selected_applied_loras = set()
        # Dictionary to store all LORA widgets
        self.all_thumbnail_widgets = {}
        
        # Load LORAs and refresh selected list
        QTimer.singleShot(0, self.load_loras)
        QTimer.singleShot(0, self.refresh_selected_list)
    
    def refresh_selected_list(self):
        """Refresh the list of selected LORAs"""
        # Clear existing thumbnails
        for i in reversed(range(self.selected_layout.count())): 
            self.selected_layout.itemAt(i).widget().setParent(None)
        if not os.path.exists(self.output_path):
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
        # Add thumbnails for each LORA
        row = 0
        col = 0
        container_width = self.selected_widget.width()
        if container_width < self.thumbnail_size + 24:
            parent = self.selected_widget.parent()
            container_width = parent.width() if parent else self.width()
        max_cols = max(1, int(container_width // (self.thumbnail_size + 24)))
        self.selected_applied_loras.clear()  # Limpiar selección al refrescar panel de abajo
        for lora_name, files in lora_files.items():
            # Try to find preview image in the original directory
            lora_dir = os.path.dirname(files['model'])
            preview_path = None
            # First try with .preview extension
            preview_base = f"{lora_name}.preview"
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(lora_dir, f"{preview_base}{ext}")
                if os.path.exists(test_path):
                    preview_path = test_path
                    break
            # If not found, try with just the LORA name
            if preview_path is None:
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(lora_dir, f"{lora_name}{ext}")
                    if os.path.exists(test_path):
                        preview_path = test_path
                        break
            # If still not found, use default
            if preview_path is None:
                preview_path = os.path.join(lora_dir, f"{lora_name}.preview.png")
                if not os.path.exists(preview_path):
                    preview_path = os.path.join(lora_dir, "preview.png")
            thumbnail_widget = self.create_thumbnail_widget(files['model'], lora_name, preview_path, is_applied=True)
            self.selected_layout.addWidget(thumbnail_widget, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
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
        except FileNotFoundError:
            self.lora_path = self.default_lora_path
            self.output_path = self.default_output_path
            self.thumbnail_size = 250
    
    def save_settings(self):
        settings = {
            'lora_path': self.lora_path,
            'output_path': self.output_path,
            'thumbnail_size': self.thumbnail_size
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)
    
    def change_lora_path(self):
        new_path = QFileDialog.getExistingDirectory(self, "Select LORA Directory", self.lora_path)
        if new_path:
            self.lora_path = new_path
            self.lora_path_label.setText(f"LORA Path: {self.lora_path}")
            self.save_settings()
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
        if rel_path != ".":
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
            # Selección por click en el título, imagen o contenedor
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
            def handle_click(event):
                current_state = lora_path in self.selected_applied_loras
                new_state = not current_state
                if new_state:
                    self.selected_applied_loras.add(lora_path)
                else:
                    self.selected_applied_loras.discard(lora_path)
                update_selection(new_state)
                self.update_remove_all_btn_text()
            image_container.mousePressEvent = handle_click
            img_label.mousePressEvent = handle_click
            name_label.mousePressEvent = handle_click
            if rel_path != ".":
                path_label.mousePressEvent = handle_click
            update_selection(lora_path in self.selected_applied_loras)
        else:
            # Panel de abajo: selección múltiple por click, sin botón X
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
            def handle_click(event):
                current_state = lora_path in self.selected_applied_loras
                new_state = not current_state
                if new_state:
                    self.selected_applied_loras.add(lora_path)
                else:
                    self.selected_applied_loras.discard(lora_path)
                update_selection(new_state)
                self.update_remove_all_btn_text()
            image_container.mousePressEvent = handle_click
            img_label.mousePressEvent = handle_click
            name_label.mousePressEvent = handle_click
            if rel_path != ".":
                path_label.mousePressEvent = handle_click
            update_selection(lora_path in self.selected_applied_loras)
        
        return thumbnail_widget
    
    def load_loras(self):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_layout.count())): 
            self.thumbnail_layout.itemAt(i).widget().setParent(None)
        self.all_thumbnail_widgets.clear()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Walk through the LORA directory
        row = 0
        col = 0
        container_width = self.thumbnail_widget.width()
        if container_width < self.thumbnail_size + 24:
            parent = self.thumbnail_widget.parent()
            container_width = parent.width() if parent else self.width()
        max_cols = max(1, int(container_width // (self.thumbnail_size + 24)))  # Solo columnas completas
        
        for root, dirs, files in os.walk(self.lora_path):
            # Find all .safetensors files
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
                        preview_path = os.path.join(root, f"{lora_name}.preview.png")  # Default preview name
                        if not os.path.exists(preview_path):
                            preview_path = os.path.join(root, "preview.png")  # Fallback preview name
                    
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
        container_width = self.thumbnail_widget.width()
        if container_width < self.thumbnail_size + 24:
            parent = self.thumbnail_widget.parent()
            container_width = parent.width() if parent else self.width()
        max_cols = max(1, int(container_width // (self.thumbnail_size + 24)))
        
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
        self.load_loras()
        self.refresh_selected_list()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoraManager()
    window.show()
    sys.exit(app.exec()) 