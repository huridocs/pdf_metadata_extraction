import pickle
from os.path import join

from config import ROOT_PATH

label_data = {
  "cejil1": [
    "Eduardo Ferrer Mac-Gregor Poisot"
  ],
  "cejil7": [
    "Sergio García Ramírez"
  ],
  "cejil8": [
    "Héctor Fix-Zamudio"
  ],
  "cejil_b1": [
    "Sergio García Ramírez"
  ],
  "cejil_b2": [
    "Rafael Nieto Navia"
  ],
  "cejil_b3": [
    "Thomas Buergenthal"
  ],
  "cejil_staging1": [
    "Diego García-Sayán"
  ],
  "cejil_staging10": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging12": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging15": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging17": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging2": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging20": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging21": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging24": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging27": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging31": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging4": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging44": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging54": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging55": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging56": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging58": [
    "L. Patricio Pazmiño Freire"
  ],
  "cejil_staging66": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging71": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging72": [
    "Elizabeth Odio Benito"
  ],
  "cejil_staging8": [
    "Joel Hernández",
    "Esmeralda E. Arosemena Bernal de Troitiño"
  ]
}

def load_paragraphs(pdf_name):
    paragraphs_path = join(ROOT_PATH, f"data/paragraphs_cache/{pdf_name}.pickle")
    print(paragraphs_path)
    with open(paragraphs_path, mode="rb") as file:
        return pickle.load(file)

if __name__ == '__main__':

    for pdf_name, presidents in label_data.items():
        load_paragraphs(pdf_name)


    print('done')

    pdf_text = "Eduardo Ferrer Mac-Gregor Poisot"
