import cv2, numpy as np

def segmentacao_parafusos(img_prepross):
    
    """ Segmenta os parafuros presentes na imagem. estiu usando otsu pq essa função calcula o limiar entre fundo e img automaticamente.

    Returns:
        _type_: _description_
    """
    
    _, binary = cv2.threshold(img_prepross, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed