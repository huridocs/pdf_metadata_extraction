import json
from pathlib import Path

import ollama

from config import ROOT_PATH

# samples = {'slander': 7, 'e-government': 7, 'right to be forgotten': 7,
#            'criminal procedure code': 7, 'domain name': 7, 'filtering and blocking': 7, 'freedom of association': 7,
#            'sedition': 7, 'encryption': 7, 'libel': 7, 'privacy, data protection and retention': 7, 'publications': 7,
#            'interception': 6, 'biometric data': 6, 'public officials': 6, 'e-transactions law': 6, 'medical ethics': 6,
#            'national security law': 6, 'radio communication': 6, 'regulatory body': 6, 'internet': 6, 'intelligence': 5,
#            'audiovisual law': 5, 'content moderation': 5, 'e-signature': 5, 'political expression': 5,
#            'e-signature law': 5, 'pornography': 5, 'isp regulation': 5, 'oversight bodies': 5, 'elections': 5,
#            'online forum': 5, 'disinformation': 5, 'internet shutdown': 5, 'consumer protection law': 4, 'religion': 4,
#            'press and media': 4, 'false news': 4, 'incitement to violence': 4, 'state of emergency': 4, 'education': 4,
#            'confidentiality': 3, 'voip': 3, 'public order': 3, 'blasphemy': 3, 'public decency': 3, 'hate speech': 3,
#            'anonymity': 3, 'administrative law': 3, 'fines': 3, 'whistleblowing': 3, 'official documents': 2,
#            'defamation ___ reputation': 2, 'digital id': 2, 'publications law': 2, 'environmental law': 2, 'morality': 2,
#            'domain name regulation': 2, 'private network regulation': 2,
#            'protection of scheduled castes and scheduled tribes': 2, 'admissibility': 2, 'child protection': 2,
#            'spam': 2, 'wiretapping': 2, 'civil code': 2, 'budget': 2, 'mobile registration': 2,
#            'open court principle': 2, 'environment': 2, 'insult to head of state': 2, 'criminal investigation': 2,
#            'civil society organizations': 2, 'sexuality': 2, 'sim registration': 2,
#            'instagram community guidelines>hate speech, bullying and abuse': 1, 'bulk messaging': 1,
#            'general regulations': 1, 'monitoring': 1, 'decryption': 1, 'publisher': 1, 'indecency ___ obscenity': 1,
#            'internet of things': 1, 'due process': 1, 'protest law': 1, 'right to fair trial': 1,
#            'freedom of association and assembly ___ protests': 1, 'incitement of hatred': 1,
#            'attorney-client privilege': 1, 'doctor ethics law': 1, 'gender': 1, 'protests': 1,
#            'representation of children': 1, 'non-consensual sharing of intimate images': 1, 'biometric': 1,
#            'metadata': 1, 'open data': 1, 'privacy policy': 1, 'regional treaty': 1, 'cyberstalking': 1,
#            'phone jamming': 1, 'contempt of court': 1, 'lgbtqi': 1, 'incitement': 1, 'contract law': 1,
#             'corruption': 1, 'cybercafe': 1, 'racism': 1, 'banking secrecy law': 1, 'fintech': 1,
#            'network disruption': 1}

samples = {
    "decryption": 1,
    "state of emergency": 1,
    "open data": 1,
    "instagram community guidelines>hate speech, bullying and abuse": 1,
    "military justice code": 1,
    "privacy policy": 1,
    "cyber security ___ cyber crime": 1,
    "gambling": 1,
    "insult to foreign state": 1,
    "cybercafe": 1,
}


def get_synthetic_data():
    for key, values in samples.items():
        path = Path(ROOT_PATH, "data", "cyrilla_synthetic_data", f"{key}.txt")

        if path.exists():
            continue

        response = ollama.chat(
            model="llama3.1:latest",
            messages=[
                {
                    "role": "user",
                    "content": f"""write eight paragraphs about government documents ruling {key} but each paragraph is tackling a different aspect""",
                }
            ],
        )

        texts = response["message"]["content"].split("\n\n")

        if len(texts) == 16:
            texts = [text for i, text in enumerate(texts) if i % 2 == 1]

        if len(texts) == 17:
            texts = [text for i, text in enumerate(texts[1:]) if i % 2 == 1]

        if len(texts) != 8:
            print(f"Error in {key}")
            break

        path.write_text(json.dumps(texts, indent=4))


if __name__ == "__main__":
    get_synthetic_data()
