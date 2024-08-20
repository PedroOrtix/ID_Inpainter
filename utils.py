import re
from prettytable import PrettyTable


def separar_cadenas(lista):
    # Inicializar el diccionario
    resultado = {
        "alpha": [],
        "numeric": []
    }
    
    # Expresiones regulares
    regex_alpha = re.compile(r'^[A-Z]+$', re.IGNORECASE)  # Solo letras
    regex_numeric = re.compile(r'^[A-Z0-9]+$', re.IGNORECASE)  # Letras y/o números
    
    # Recorrer cada elemento de la lista
    for elemento in lista:
        if regex_alpha.match(elemento):
            resultado["alpha"].append(elemento)
        elif regex_numeric.match(elemento):
            resultado["numeric"].append(elemento)
    
    return resultado

def mostrar_diccionario_ascii(diccionario):
    # Crear una tabla
    tabla = PrettyTable()

    # Establecer los encabezados de las columnas
    tabla.field_names = ["Clave", "Valores"]

    # Iterar sobre el diccionario y agregar las filas a la tabla
    for clave, valores in diccionario.items():
        valores_formateados = "\n".join(valores)
        tabla.add_row([clave, valores_formateados])

    # Mostrar la tabla en formato ASCII
    print(tabla)