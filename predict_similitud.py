#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_top3.py

Carga artefacto, preprocesa asuntos (lower+unidecode), calcula embeddings
y extrae Top‑3 distintos, aplica umbral para no categorizados.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

# Preprocess text

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = unidecode(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description='Top-3 similitud con umbral y preprocessing')
    parser.add_argument('artefacto', help='artefacto_similitud.joblib')
    parser.add_argument('asuntos_excel', help='asuntos diarios .xlsx')
    parser.add_argument('--threshold', '-t', type=float, default=0.25,
                        help='Umbral mínimo para categorizar')
    parser.add_argument('--output', '-o', default='resultados_produccion.xlsx',
                        help='Excel de salida')
    args = parser.parse_args()

    arte = joblib.load(args.artefacto)
    model_name = arte['model_name']
    meta_df    = arte['meta_df']
    syn_emb    = arte['syn_emb']

    df_as = pd.read_excel(args.asuntos_excel)
    texts = [preprocess(str(x)) for x in df_as['Asunto']]

    model = SentenceTransformer(model_name)
    as_emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    sim_mat = cosine_similarity(as_emb, syn_emb)
    topk = 3
    idx_sorted = np.argsort(-sim_mat, axis=1)[:, :topk]
    top_vals   = np.take_along_axis(sim_mat, idx_sorted, axis=1)

    out = pd.DataFrame({
        'Asunto': df_as['Asunto'].astype(str).tolist(),
        **({'ID': df_as['ID']}     if 'ID' in df_as else {}),
        **({'Correo': df_as['Correo']} if 'Correo' in df_as else {})
    })

    for i in range(topk):
        col_type = f'Tipo_{i+1}'
        col_ramo = f'Ramo_{i+1}'
        col_sim  = f'Sim_{i+1}'
        col_fila = f'Fila_{i+1}'
        out[col_type] = meta_df['Tipo de Movimiento'].iloc[idx_sorted[:, i]].values
        out[col_ramo] = meta_df['Ramo'].iloc[idx_sorted[:, i]].values
        # Aplicar umbral: si sim < threshold, marcar NO_CATEGORIZADO
        sims = top_vals[:, i]
        pct = [f"{v*100:.2f}%" for v in sims]
        # Plantilla: si primer match < umbral, solo el primer match
        if i == 0:
            out[col_type] = [('NO_CATEGORIZADO' if v < args.threshold else t)
                              for v, t in zip(sims, out[col_type])]
        out[col_sim]  = pct
        out[col_fila] = meta_df['Fila'].iloc[idx_sorted[:, i]].values

    out.to_excel(args.output, index=False)
    print(f"[+] Resultados guardados en: {args.output}")

if __name__ == '__main__':
    main()
