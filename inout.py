"""
Funções de IO com as imagens.
"""
import cv2
import numpy as np

# anotações de tipo para 3.7+
from sys import version_info
if version_info.minor >= 7:
    from tipos import Image
else: # para 3.6
    Image = 'Image' # type: ignore


def imgread(arquivo: str, modo: int=cv2.IMREAD_COLOR) -> Image:
    """
    Lê um arquivo de imagem em escala de cinza.

    Parâmetros
    ----------
    arquivo: str
        Caminho para o arquivo de imagem a ser lido.

    Retorno
    -------
    img: np.ndarray
        Matriz representando a imagem lida.
    """
    # abre o arquivo fora do OpenCV, para que o
    # Python trate os erros de IO
    with open(arquivo, mode='rb') as filebuf:
        buf = np.frombuffer(filebuf.read(), dtype=np.uint8)

    # só resta tratar problemas de decodificação
    img: Image = cv2.imdecode(buf, modo)
    if img is None:
        msg = f'não foi possível parsear "{arquivo}" como imagem'
        raise ValueError(msg)

    return img


def imgwrite(img: Image, arquivo: str) -> None:
    """
    Escreve uma matriz como imagem PNG em um arquivo.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz representando uma imagem.
    arquivo: str
        Caminho para o arquivo onde a imagem será gravada.

    Erro
    ----
    ValueError
        Quando a imagem não pode ser salva no arquivo ou quando
        a entrada não representa uma imagem ou não pode ser
        convertido para a extensão eserada.
    """
    # indica para o caller quando a imagem NÃO for salva
    if not cv2.imwrite(arquivo, img):
        msg = f'não foi possível salvar a imagem em "{arquivo}"'
        raise ValueError(msg)


def imgshow(img: Image, nome: str="") -> None:
    """
    Apresenta a imagem em uma janela com um nome.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz representando uma imagem.
    nome: str
        Nome da janela a ser aberta. Opcional.
    """
    try:
        cv2.imshow(nome, img)
        cv2.waitKey()
    # Ctrl-C não são erros nesse caso
    except KeyboardInterrupt:
        pass
