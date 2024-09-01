# file_utils.py
import os
import shutil
import datetime
import uuid

def create_directory(directory_path: str) -> None:
    """
    Crea un directorio si no existe.

    Args:
        directory_path (str): Ruta del directorio a crear.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def delete_directory(directory_path: str) -> None:
    """
    Elimina un directorio y su contenido.

    Args:
        directory_path (str): Ruta del directorio a eliminar.

    Raises:
        FileNotFoundError: Si el directorio no existe.
        PermissionError: Si no se tienen permisos para eliminar el directorio.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    try:
        shutil.rmtree(directory_path)
    except PermissionError:
        raise PermissionError(f"Permission denied: Unable to delete '{directory_path}'.")

def generate_unique_name():
    """
    Genera un nombre único basado en la fecha, hora actual y un UUID.

    Returns:
        str: Nombre único generado.
    """
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4()
    return f"{current_datetime}_{unique_id}"