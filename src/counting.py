import cv2

def contagem_parafusos(img_segmentada, imagem):
    """ Função que conta os parafusos a partir da imagem segmanetada

    Args:
        img_segmentada (_type_): _description_
        imagem (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    output = imagem.copy()
    
    contornos = cv2.findContours(img_segmentada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parafusos = 0
    
    detections = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 100: 
            continue
            
        x,y,w,h = cv2.boundingRect(c)
        
        if h == 0:
            continue
        
        aspect_ratio = w/h
        if not (0.15 <= aspect_ratio <= 6.0):
            continue
        
        parafusos +=1
        detections.append({
            'id': parafusos,
            'x': x,
            'y': y,
            'height': h, 
            'area': area,
            'aspect_ration': aspect_ratio
        })
        
        cv2.rectangle(output, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output, str(parafusos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return parafusos, output, detections