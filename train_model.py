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

def load_synonyms(excel_path: str):
    df = pd.read_excel(excel_path)
    # Metadatos
    meta = df[['Tipo de Movimiento', 'Ramo']].copy()
    # Columnas de opciones
    opt_cols = [c for c in df.columns if 'Opcion' in c]
    syn_texts = []
    for _, row in df.iterrows():
        # Concatenamos 'Tipo de Movimiento' y 'Ramo' al texto base
        base = f"{row['Tipo de Movimiento']} {row['Ramo']}"
        opts = [str(row[c]).strip() for c in opt_cols
                if pd.notna(row[c]) and str(row[c]).strip()]
        text = ' | '.join([base] + opts)
        syn_texts.append(text)
    return meta, syn_texts

def main():
    parser = argparse.ArgumentParser(
        description='Entrena artefacto de embeddings de sinónimos (2000 filas).')
    parser.add_argument('synonyms_excel',
                        help='Catalogo de movimientos (.xlsx)')
    parser.add_argument('--output', '-o', default='artefacto_similitud.joblib',
                        help='Ruta de salida .joblib')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                        help='Modelo SBERT para embeddings')
    args = parser.parse_args()

    # Carga y prepara textos
    meta_df, syn_texts = load_synonyms(args.synonyms_excel)

    # Entrena embeddings
    model = SentenceTransformer(args.model)
    syn_emb = model.encode(
        syn_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Guarda artefacto
    artefact = {
        'model_name': args.model,
        'meta_df': meta_df,
        'syn_texts': syn_texts,
        'syn_emb': syn_emb
    }
    joblib.dump(artefact, args.output)
    print(f"[+] Artefacto guardado en: {args.output}")

if __name__ == '__main__':
    main()