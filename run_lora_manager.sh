#!/bin/bash

# Nombre del entorno virtual
VENV_NAME="venv"

# Función para mostrar mensajes
print_message() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Verificar si Python 3.10 está instalado
if ! command -v python3.10 &> /dev/null; then
    print_message "Python 3.10 no está instalado. Por favor, instala Python 3.10 primero."
    print_message "En Ubuntu/Debian puedes instalarlo con: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

# Verificar si pip está instalado
if ! command -v pip3 &> /dev/null; then
    print_message "pip3 no está instalado. Por favor, instala pip3 primero."
    exit 1
fi

# Instalar dependencias del sistema para Qt
print_message "Instalando dependencias del sistema para Qt..."
sudo apt-get update
sudo apt-get install -y python3-venv qt6-base-dev libqt6gui6 libqt6widgets6 libqt6core6 libqt6svg6

# Crear entorno virtual si no existe
if [ ! -d "$VENV_NAME" ]; then
    print_message "Creando entorno virtual con Python 3.10..."
    python3.10 -m venv $VENV_NAME
fi

# Activar el entorno virtual
print_message "Activando entorno virtual..."
source $VENV_NAME/bin/activate

# Verificar que el entorno virtual está activado
if [ -z "$VIRTUAL_ENV" ]; then
    print_message "ERROR: El entorno virtual no se ha activado correctamente"
    exit 1
else
    print_message "Entorno virtual activado en: $VIRTUAL_ENV"
fi

# Instalar/actualizar pip
print_message "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
print_message "Instalando dependencias..."
pip install -r requirements.txt

# Ejecutar el programa
print_message "Iniciando LORA Manager..."
python lora_manager.py

# Desactivar el entorno virtual al salir
deactivate 