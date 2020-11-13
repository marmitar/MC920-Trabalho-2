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


def imgshow(img: Image, nome: str="", delay: int=250) -> None:
    """
    Apresenta a imagem em uma janela com um nome.

    Parâmetros
    ----------
    img: np.ndarray
        Matriz representando uma imagem.
    nome: str, opcional
        Nome da janela a ser aberta.
    delay: int, opcional
        Tempo em milisegundos de checagem da janela.
    """
    try:
        cv2.namedWindow(nome, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(nome, img)

        # espera alguma chave ou a janela ser fechada
        while cv2.waitKey(delay) < 0:
            # problemas com versões diferentes de python e opencv
            prop1 = cv2.getWindowProperty(nome, cv2.WND_PROP_ASPECT_RATIO)
            prop2 = cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE)
            if prop1 == prop2:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    # Ctrl-C não são erros nesse caso
    except KeyboardInterrupt:
        pass
