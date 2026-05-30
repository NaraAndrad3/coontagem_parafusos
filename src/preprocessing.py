import cv2

def preprocessamento_imagem(img):
    """ Prepara a imagem para a etapa de segmentação. 
    

    Args:
        img: imagem a ser processada

    Returns:
        blurred: imagem preprocessada
    """
    
    # cobverte para a escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # melhora o contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    enhanced = clahe.apply(gray)
    # suaviza os ruidos da imagem com o filtro guassiano.
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    return blurred