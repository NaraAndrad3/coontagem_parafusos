import cv2, numpy as np

class Parafusos:
    def __init__(self, min_area = 100, max_area = 50000, min_aspect_ratio = 0.15, max_aspect_ratio = 6.0, blur_kernel = (5,5)):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ration = max_aspect_ratio
        self.blur_kernel = blur_kernel
    
    def carrega_imagem(self, path_image):
        image = cv2.imread(path_image)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {path_image}")
        
        return image
    
    def preprocessa_imagem(self, image):
        """ Prepara a imagem para a etapa de segmentação. 
    

        Args:
            img: imagem a ser processada

        Returns:
            blurred: imagem preprocessada
        """
        
        # cobverte para a escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # melhora o contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        enhanced = clahe.apply(gray)
        # suaviza os ruidos da imagem com o filtro guassiano.
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        
        return blurred
    
    def segmenta_imagem(self, img):
        
        """ Segmenta os parafuros presentes na imagem. estiu usando otsu pq essa função calcula o limiar entre fundo e img automaticamente.

        Returns:
            _type_: _description_
        """
        
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed
    
    def area_valida_parafuso(self, contorno):
        """  Decide se uma região segmentada deve ser considerada parafuso.

        Args:
            contorno (_type_): contorno
        """
        
        area = cv2.contourArea(contorno)
        if area < self.min_area or area > self.max_area:
            return False
        
        x,y,w,h = cv2.boundingRect(contorno)
        
        if h==0:
            return False
        aspect_ratio = w/h
        
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ration):
            return False
        return True
    
    
        
        
        
