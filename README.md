# Projeto de Reconhecimento de Lesões de Pele com CNN e API Flask

## 1. Descrição do Projeto

Este projeto utiliza uma rede neural convolucional (CNN) para classificar imagens de lesões de pele em diferentes categorias dermatológicas, utilizando o dataset **Dermamnist** do MedMNIST. Além disso, disponibiliza uma API RESTful construída com Flask para realizar predições a partir de imagens enviadas pelo usuário.

---

## 2. Classes Reconhecidas (Tipos de Lesões)

O modelo reconhece as seguintes classes:

| Classe | Descrição                   |
|--------|-----------------------------|
| bkl    | Queratoses benignas         |
| nv     | Nevos (pintas, sinais)      |
| mel    | Melanoma (tumor maligno)    |
| df     | Dermatofibroma              |
| vasc   | Lesões vasculares           |
| akiec  | Queratose actínica          |
| scc    | Carcinoma de células escamosas |

---

## 3. Requisitos

- Python 3.8 ou superior
- pip
- Ambiente virtual (recomendado)

---

## 4. Instalação

1. Clone este repositório ou baixe os arquivos do projeto.
2. Crie e ative um ambiente virtual:

```bash
python -m venv venv
# No Windows
venv\Scripts\activate
# No Linux/macOS
source venv/bin/activate
