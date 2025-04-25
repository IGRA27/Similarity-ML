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
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

def load_synonyms(excel_path: str):
    df = pd.read_excel(excel_path)
    # Columnas de metadatos
    meta = df[['Tipo de Movimiento', 'Ramo']].copy()
    # Columnas de opciones
    opt_cols = [c for c in df.columns if 'Opcion' in c]
    syn_texts = []
    for _, row in df.iterrows():
        opts = [str(row[c]).strip()
                for c in opt_cols
                if pd.notna(row[c]) and str(row[c]).strip()]
        syn_texts.append(' | '.join(opts))
    return meta, syn_texts

def main():
    parser = argparse.ArgumentParser(
        description='Entrena y guarda artefacto de embeddings de sinónimos.')
    parser.add_argument('synonyms_excel', help='Excel de sinónimos')
    parser.add_argument('--output', '-o', default='artefacto_similitud.joblib',
                        help='Fichero joblib de salida')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                        help='Modelo SBERT a usar')
    args = parser.parse_args()

    # 1) Cargo y preparo textos
    meta_df, syn_texts = load_synonyms(args.synonyms_excel)

    # 2) Cargo SBERT y cálculo embeddings
    model = SentenceTransformer(args.model)
    syn_emb = model.encode(syn_texts,
                           batch_size=64,
                           show_progress_bar=True,
                           convert_to_numpy=True)

    # 3) Creo artefacto
    artefact = {
        'model_name': args.model,
        'meta_df': meta_df,           # DataFrame con Tipo de Movimiento y Ramo
        'syn_texts': syn_texts,       # lista de strings
        'syn_emb': syn_emb            # numpy array (N_categoria × dim_emb)
    }

    # 4) Guardo con joblib
    joblib.dump(artefact, args.output)
    print(f"[+] Artefacto guardado en: {args.output}")

if __name__ == '__main__':
    main()
