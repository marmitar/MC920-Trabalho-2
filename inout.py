"""
Funções de IO com as imagens.
"""
from tipos import Image
import cv2
import numpy as np


def imgread(arquivo: str, modo: int=cv2.IMREAD_COLOR) -> Image:
    """
    Lê um arquivo de imagem em escala de cinza.

    Parâmetros
    ----------
    arquivo: str
        Caminho para o arquivo de imagem a ser lido.
    modo: int
        Flag de leitura da biblioteca OpenCV.

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
