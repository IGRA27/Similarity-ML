#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    p = argparse.ArgumentParser(
        description='Inferencia de similitud con artefacto pre-entrenado')
    p.add_argument('artefacto',
                   help='Ruta al archivo artefacto_similitud.joblib')
    p.add_argument('asuntos_excel',
                   help='Excel con columna "Asunto" (y opcionalmente "ID" y "Correo")')
    p.add_argument('--output', '-o',
                   default='resultados_produccion.xlsx',
                   help='Excel de salida con resultados')
    args = p.parse_args()

    # 1) Carga artefacto previamente generado
    arte = joblib.load(args.artefacto)
    model_name = arte['model_name']
    meta_df    = arte['meta_df']
    syn_emb    = arte['syn_emb']
    syn_texts  = arte['syn_texts']

    # 2) Carga el Excel de asuntos
    df_as = pd.read_excel(args.asuntos_excel)
    texts = df_as['Asunto'].astype(str).tolist()

    # 3) Calcula embeddings de los asuntos
    model = SentenceTransformer(model_name)
    as_emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 4) Calcula similitud coseno y extrae mejor categoría
    sim_mat  = cosine_similarity(as_emb, syn_emb)
    best_idx = sim_mat.argmax(axis=1)
    # ==> aquí formateamos el porcentaje en una lista de strings
    best_sims_numeric = (sim_mat.max(axis=1) * 100).round(2)
    best_sim = [f"{v:.2f}%" for v in best_sims_numeric]

    # 5) Prepara DataFrame de salida
    out = pd.DataFrame({
        'Asunto': texts,
        **({'ID': df_as['ID']}         if 'ID' in df_as else {}),
        **({'Correo': df_as['Correo']} if 'Correo' in df_as else {}),
        'Tipo de Movimiento': meta_df['Tipo de Movimiento']
                                 .iloc[best_idx].values,
        'Ramo':                meta_df['Ramo']
                                 .iloc[best_idx].values,
        'Categoría Ejemplo':   [syn_texts[i] for i in best_idx],
        'Similitud':           best_sim
    })

    # 6) Exporta a Excel
    out.to_excel(args.output, index=False)
    print(f"[+] Resultados guardados en: {args.output}")

if __name__ == '__main__':
    main()
