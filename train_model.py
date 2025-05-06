#AUTOR: ISAAC REYES
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py

Entrena una “base” de embeddings de categorías (sinónimos) y guarda un artefacto para inferencia.

Uso:
  python train_model.py \
    /ruta/catalogo_movimientos.xlsx \
    --output artefacto_similitud.joblib \
    --model all-MiniLM-L6-v2
Entrena y guarda un artefacto de embeddings de categorías de sinónimos.
"""
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from unidecode import unidecode

#Text preprocessing: lowercase, remove accents, strip

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = unidecode(text)
    return text


def load_and_dedupe(excel_path: str):
    df = pd.read_excel(excel_path)
    df['Fila'] = df.index + 2
    opt_cols = [c for c in df.columns if c.lower().startswith('opcion')]
    syn_texts = []
    for _, row in df.iterrows():
        base = f"{row['Tipo de Movimiento']} {row['Ramo']}"
        opts = [str(row[c]).strip() for c in opt_cols
                if pd.notna(row[c]) and str(row[c]).strip()]
        full = ' | '.join([base] + opts)
        syn_texts.append(preprocess(full))
    df['Syn_Text'] = syn_texts
    dedup = df.drop_duplicates(subset='Syn_Text', keep='first').reset_index(drop=True)
    meta = dedup[['Tipo de Movimiento', 'Ramo', 'Fila']].copy()
    return meta, dedup['Syn_Text'].tolist()

#Nota Isaac: agregue un mejor modelo, mejor en español importante
#Se puede hacer fine-tunning para mejorar.
def main():
    parser = argparse.ArgumentParser(
        description='Entrena artefacto SBERT de sinónimos.')
    parser.add_argument('synonyms_excel',
                        help='catalogo_movimientos.xlsx')
    parser.add_argument('--output', '-o', default='artefacto_similitud.joblib',
                        help='joblib de artefacto')
    parser.add_argument('--model', '-m',
                        default='paraphrase-multilingual-mpnet-base-v2',
                        help='Modelo SBERT para embeddings')
    args = parser.parse_args()

    meta_df, syn_texts = load_and_dedupe(args.synonyms_excel)
    model = SentenceTransformer(args.model)
    syn_emb = model.encode(
        syn_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    arte = {
        'model_name': args.model,
        'meta_df': meta_df,
        'syn_emb': syn_emb
    }
    joblib.dump(arte, args.output)
    print(f"[+] Artefacto guardado en: {args.output}")

if __name__ == '__main__':
    main()
