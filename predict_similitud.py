#!/usr/bin/env python3
"""
Inferencia de similitud: extrae top-3 matches, porcentajes y fila.
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser(
        description='Top-3 similitud entre asuntos y catálogo único')
    parser.add_argument('artefacto', help='artefacto_similitud.joblib')
    parser.add_argument('asuntos_excel', help='asuntos diarios (.xlsx)')
    parser.add_argument('--output', '-o', default='resultados_produccion.xlsx',
                        help='Excel de salida')
    args = parser.parse_args()

    #1)Cargo artefacto SBERT + meta únicos
    arte = joblib.load(args.artefacto)
    model_name = arte['model_name']
    meta_df    = arte['meta_df']  # cols: Tipo de Movimiento, Ramo, Fila
    syn_emb    = arte['syn_emb']

    #2)Cargo asuntos
    df_as = pd.read_excel(args.asuntos_excel)
    texts = df_as['Asunto'].astype(str).tolist()

    #3)Embedding de asuntos
    model = SentenceTransformer(model_name)
    as_emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    #4)Similitud coseno + top-3 índices
    sim_mat = cosine_similarity(as_emb, syn_emb)
    topk = 3
    idx_sorted = np.argsort(-sim_mat, axis=1)[:, :topk]
    top_vals  = np.take_along_axis(sim_mat, idx_sorted, axis=1)

    #5)Preparo resultados
    out = pd.DataFrame({
        'Asunto': texts,
        **({'ID': df_as['ID']}     if 'ID' in df_as else {}),
        **({'Correo': df_as['Correo']} if 'Correo' in df_as else {})
    })
    for i in range(topk):
        out[f'Tipo_{i+1}'] = meta_df['Tipo de Movimiento'].iloc[idx_sorted[:, i]].values
        out[f'Ramo_{i+1}'] = meta_df['Ramo'].iloc[idx_sorted[:, i]].values
        out[f'Sim_{i+1}']  = [f"{v*100:.2f}%" for v in top_vals[:, i]]
        out[f'Fila_{i+1}'] = meta_df['Fila'].iloc[idx_sorted[:, i]].values

    #6)Exportamos
    out.to_excel(args.output, index=False)
    print(f"[+] Resultados guardados en: {args.output}")

if __name__ == '__main__':
    main()
