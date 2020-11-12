import sys, cv2, warnings
from argparse import ArgumentParser, ArgumentTypeError

from inout import imgread, imgwrite, imgshow
from lib import meios_tons, Varredura, USANDO_NUMBA
from dists import ERR_DIST, ErrorDist


if sys.version_info.major < 3 or sys.version_info.minor < 8:
    msg = """

    Essa ferramenta foi desenvolvida com Python 3.8 e funciona corretamente
    nessa versão. Python 3.7 deveria funcionar sem erros também. Python 3.6
    funciona, mas com algumas funcionalidades limitadas. Outras versões não
    funcionam por causa das f-strings.
    """
    warnings.warn(msg)
if not USANDO_NUMBA:
    msg = """

    Sem as otimizações do pacote Numba (https://numba.pydata.org/) a ferramenta
    demora cerca de 30 vezes mais do que o necessário.
    """
    warnings.warn(msg)


# # # # # # # # # # # # # # #
# Tratamento dos argumentos #

def dist_err(nome: str) -> ErrorDist:
    """
    Recuperação das distribuições de erro pré-definidas.
    """
    try:
        return ERR_DIST[nome]
    except KeyError:
        msg = f'distribuição de erro inválida: {nome}'
        raise ArgumentTypeError(msg)

def varredura(nome: str) -> Varredura:
    """
    Processamento dos argumentos de varredura da imagem.
    """
    try:
        return Varredura[nome]
    except KeyError:
        msg = f'opção de varredura inválida: {nome}'
        raise ArgumentTypeError(msg)


# parser de argumentos
description = 'Ferramenta de aplicação de meios-tons para o Trabalho 2.'
usage = '%(prog)s [OPTIONS] INPUT'

parser = ArgumentParser(description=description, usage=usage, allow_abbrev=False)
# argumentos necessários
parser.add_argument('input', metavar='INPUT', type=str,
                    help='imagem de entrada')
# opções de saída
parser.add_argument('-o', '--output', type=str, action='append', metavar='FILE',
                    help='arquivo para gravar o resultado')
parser.add_argument('-f', '--force-show', action='store_true',
                    help='sempre mostra o resultado final em uma janela')
# configurações do pontilhado
parser.add_argument('-g', '--grayscale', dest='modo', action='store_const',
                    default=cv2.IMREAD_COLOR, const=cv2.IMREAD_GRAYSCALE,
                    help='aplica meios-tons em imagem escala de cinza')
parser.add_argument('-v', '--varredura',
                    type=varredura, choices=Varredura, default=Varredura.alternada,
                    help='muda a forma de varredura da imagem (PADRÃO: alternada)')
parser.add_argument('-d', '--distribuicao', dest='dist', type=dist_err, default='FLOYD_STEINBERG',
                    help='muda a distribuição de erros do pontilhado (PADRÃO: FLOYD_STEINBERG)')


if __name__ == "__main__":
    # argumentos fora de ordem no Python 3.7+
    if sys.version_info.minor >= 7:
        args = parser.parse_intermixed_args()
    else:
        args = parser.parse_args()

    # entrada
    arquivo = args.input
    img = imgread(arquivo, args.modo)

    # aplica pontilhado
    img = meios_tons(img, args.dist, args.varredura)
    # range completo para a visualização
    img *= 255

    # saída
    if args.output:
        for output in args.output:
            try:
                imgwrite(img, output)
            # em caso de erro, mostra o erro
            # mas continua a execução
            except ValueError as err:
                print(err, file=sys.stderr)

    if args.output is None or args.force_show:
        imgshow(img, arquivo)
