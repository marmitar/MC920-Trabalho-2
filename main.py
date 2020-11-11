from tipos import ErrorDist, Image
from typing import List

import numpy as np
import cv2
from lib import meios_tons



def distribuicao(data: List[List[float]]) -> ErrorDist:
    return np.asarray(data, dtype=np.float32, order='C')


FLOYD = distribuicao([
    [   0,    0, 7/16],
    [3/16, 5/16, 1/16]
])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'imagens/lenna.png'

    img = cv2.imread(filename)
    img = 255 * meios_tons(img, FLOYD, varredura='alternada')

    cv2.imwrite('out.png', img)
    # cv2.imshow(filename, img)
    # cv2.waitKey()
