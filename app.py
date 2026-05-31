"""
=============================================================
 SISTEMA DE CONTAGEM DE PARAFUSOS
 Interface Web com Streamlit
=============================================================

Autor: Nara Dias
Bootcamp de Ciência de Dados e IA

Descrição:
Aplicação web para contagem automática de parafusos
utilizando Visão Computacional Clássica com OpenCV.

Funcionalidades:
- Upload de imagem
- Contagem automática
- Visualização do pipeline completo
- Exibição da imagem anotada
- Métricas da detecção
=============================================================
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.detector import process_uploaded_image


# ------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------

st.set_page_config(
    page_title="Contador de Parafusos",
    layout="wide"
)


# ------------------------------------------------------------
# CABEÇALHO
# ------------------------------------------------------------

st.title(" >> Sistema Inteligente de Contagem de Parafusos")

st.markdown(
    """
Sistema desenvolvido utilizando **Visão Computacional Clássica**
para automatizar a contagem de parafusos em processos de picking.

O algoritmo utiliza:

- OpenCV
- Binarização automática por Otsu
- Operações Morfológicas
- Análise de Contornos
- Watershed para objetos sobrepostos
"""
)

st.divider()


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Selecione uma imagem",
    type=["jpg", "jpeg", "png"]
)


# ------------------------------------------------------------
# PROCESSAMENTO
# ------------------------------------------------------------

if uploaded_file is not None:

    image_pil = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image_pil)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    with st.spinner("Processando imagem..."):

        result = process_uploaded_image(image_bgr)

    st.success("Processamento concluído com sucesso!")

    st.divider()

    # --------------------------------------------------------
    # RESULTADO PRINCIPAL
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Imagem Original")

        st.image(
            image_pil,
            use_container_width=True
        )

    with col2:

        st.subheader("Resultado da Detecção")

        result_rgb = cv2.cvtColor(
            result["result_img"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    st.subheader("Métricas")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Parafusos",
            result["count"]
        )

    with m2:
        st.metric(
            "Corpos Detectados",
            result["body_count"]
        )

    with m3:
        st.metric(
            "Cabeças Detectadas",
            result["head_count"]
        )

    with m4:
        st.metric(
            "Blobs Válidos",
            result["blobs"]
        )

    st.divider()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader("Pipeline de Processamento")

    stages = result["stages"]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.caption("1. Original")

        original_rgb = cv2.cvtColor(
            stages["original"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            use_container_width=True
        )

    with c2:

        st.caption("2. Gaussian Blur")

        processed_rgb = cv2.cvtColor(
            stages["processed"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            processed_rgb,
            use_container_width=True
        )

    with c3:

        st.caption("3. Binarização Otsu")

        st.image(
            stages["binary"],
            use_container_width=True,
            clamp=True
        )

    with c4:

        st.caption("4. Morfologia")

        st.image(
            stages["mask_clean"],
            use_container_width=True,
            clamp=True
        )

    with c5:

        st.caption("5. Resultado")

        result_rgb = cv2.cvtColor(
            stages["result"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            use_container_width=True
        )

    st.divider()


    with st.expander(
        " >> Como o algoritmo chegou nesse resultado?"
    ):

        st.markdown(
            """
            ### Etapa 1: Pré-processamento
            A imagem recebe suavização Gaussiana para reduzir ruídos.

            ### Etapa 2 : Binarização
            Utiliza o método de Otsu para separar automaticamente
            objetos e fundo.

            ### Etapa 3 : Morfologia
            Remove ruídos e pequenas imperfeições da máscara.

            ### Etapa 4 : Extração de Contornos
            Identifica regiões candidatas a parafusos.

            ### Etapa 5 : Regras Geométricas
            São avaliadas características como:

            - Área
            - Razão de aspecto
            - Solidity
            - Extent

            ### Etapa 6 : Watershed
            Quando vários parafusos aparecem agrupados,
            o algoritmo estima quantos objetos existem
            dentro do mesmo contorno.

            ### Etapa 7 : Fusão de Evidências
            A contagem final considera:

            - Corpos detectados
            - Cabeças detectadas

            Selecionando a estimativa mais confiável.
            """
        )

else:

    st.info(
        "Envie uma imagem para iniciar a contagem."
    )

    st.image(
        "https://www.daazrolamentos.com/paraf-allen-ccl-6x8",
        caption="Exemplo de imagem para teste"
    )